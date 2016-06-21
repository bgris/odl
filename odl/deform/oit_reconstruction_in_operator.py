# Copyright 2014-2016 The ODL development group
#
# This file is part of ODL.
#
# ODL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ODL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ODL.  If not, see <http://www.gnu.org/licenses/>.

"""
Example of shape-based image reconstruction
using optimal information transformation.
"""

# Initial setup
import numpy as np
import numba
import matplotlib.pyplot as plt
import time
import ddmatch
from numbers import Number
from odl.operator.operator import Operator, OperatorComp
import odl


class Functional(Operator):

    """Quick hack for a functional class."""

    def __init__(self, domain, linear=False):
        """Initialize a new instance.

        Parameters
        ----------
        domain : `LinearSpace`
            Set of elements on which the functional can be evaluated
        linear : `bool`, optional
            If `True`, assume that the functional is linear
        """
        super().__init__(domain=domain, range=domain.field, linear=linear)

    def gradient(self, x, out=None):
        """Evaluate the gradient of the functional.

        Parameters
        ----------
        x : domain element-like
            Point in which to evaluate the gradient
        out : domain element, optional
            Element into which the result is written

        Returns
        -------
        out : domain element
            Result of the gradient calcuation. If ``out`` was given,
            the returned object is a reference to it.
        """
        raise NotImplementedError

    def __mul__(self, other):
        """Return ``self * other``.

        If ``other`` is an operator, this corresponds to
        operator composition:

            ``op1 * op2 <==> (x --> op1(op2(x))``

        If ``other`` is a scalar, this corresponds to right
        multiplication of scalars with operators:

            ``op * scalar <==> (x --> op(scalar * x))``

        If ``other`` is a vector, this corresponds to right
        multiplication of vectors with operators:

            ``op * vector <==> (x --> op(vector * x))``

        Note that left and right multiplications are generally
        different.

        Parameters
        ----------
        other : {`Operator`, `LinearSpaceVector`, scalar}
            `Operator`:
                The `Operator.domain` of ``other`` must match this
                operator's `Operator.range`.

            `LinearSpaceVector`:
                ``other`` must be an element of this operator's
                `Operator.domain`.

            scalar:
                The `Operator.domain` of this operator must be a
                `LinearSpace` and ``other`` must be an
                element of the ``field`` of this operator's
                `Operator.domain`.

        Returns
        -------
        mul : `Functional`
            Multiplication result

            If ``other`` is an `Operator`, ``mul`` is a
            `FunctionalComp`.

            If ``other`` is a scalar, ``mul`` is a
            `FunctionalRightScalarMult`.

            If ``other`` is a vector, ``mul`` is a
            `FunctionalRightVectorMult`.

        """
        if isinstance(other, Operator):
            return FunctionalComp(self, other)
        elif isinstance(other, Number):
            # Left multiplication is more efficient, so we can use this in the
            # case of linear operator.
            raise NotImplementedError
            if self.is_linear:
                return OperatorLeftScalarMult(self, other)
            else:
                return OperatorRightScalarMult(self, other)
        elif isinstance(other, LinearSpaceVector) and other in self.domain:
            raise NotImplementedError
            return OperatorRightVectorMult(self, other.copy())
        else:
            return NotImplemented


class FunctionalComp(Functional, OperatorComp):

    """Composition of a functional with an operator."""

    def __init__(self, func, op, tmp1=None, tmp2=None):
        """Initialize a new instance.

        Parameters
        ----------
        func : `Functional`
            The left ("outer") operator
        op : `Operator`
            The right ("inner") operator. Its range must coincide with the
            domain of ``func``.
        tmp1 : `element` of the range of ``op``, optional
            Used to avoid the creation of a temporary when applying ``op``
        tmp2 : `element` of the range of ``op``, optional
            Used to avoid the creation of a temporary when applying the
            gradient of ``func``
        """
        if not isinstance(func, Functional):
            raise TypeError('functional {!r} is not a Functional instance.'
                            ''.format(func))

        OperatorComp.__init__(self, left=func, right=op, tmp=tmp1)

        if tmp2 is not None and tmp2 not in self._left.domain:
            raise TypeError('second temporary {!r} not in the domain '
                            '{!r} of the functional.'
                            ''.format(tmp2, self._left.domain))
        self._tmp2 = tmp2

    def gradient(self, x, out=None):
        """Gradient of the compositon according to the chain rule.

        Parameters
        ----------
        x : domain element-like
            Point in which to evaluate the gradient
        out : domain element, optional
            Element into which the result is written

        Returns
        -------
        out : domain element
            Result of the gradient calcuation. If ``out`` was given,
            the returned object is a reference to it.
        """
        if out is None:
            # adj_op = self._right.derivative(x).adjoint
            # return adj_op(self._left.gradient(self._right(x)))
            return self._right.derivative(x).adjoint(
                self._left.gradient(self._right(x)))
        else:
            if self._tmp is not None:
                tmp_op_ran = self._right(x, out=self._tmp)
            else:
                tmp_op_ran = self._right(x)

            if self._tmp2 is not None:
                tmp_dom = self._left.gradient(tmp_op_ran, out=self._tmp2)
            else:
                tmp_dom = self._left.gradient(tmp_op_ran)

            self._right.derivative(x).adjoint(tmp_dom, out=out)


class MPDeformationOperator(Operator):
    """
    Mass-preserving deformation operator
    mapping parameter to a fixed deformed template.

    This operator computes for a fixed template ``I`` the deformed
    template::

        invphi(.) --> detjacinvphi * I(invphi(.))

    where ``invphi`` is the deformation parameter as follows::

        invphi: x --> invphi(x)

    Here, ``x`` is an element in the domain of target (ground truth).
    """

    def __init__(self, template):
        """Initialize a new instance.

        Parameters
        ----------
        template : `DiscreteLpVector`
            Fixed template deformed by the vector field. Its space
            must have the same number of dimensions as ``par_space``.
        """
        self.template = template

        # Create the space for the inverse deformation
        self.domain_space = odl.ProductSpace(self.template.space,
                                             self.template.space.ndim)

        super().__init__(self.domain_space, self.template.space, linear=False)

    def _call(self, invphi, detjacinvphi):
        """Implementation of ``self(invphi, detjacinvphi)``.

        Parameters
        ----------
        invphi: `ProductSpaceVector`
            General inverse deformation for image grid points.
        detjacinvphi: 'DiscreteLpVector'
            The determinant of the inverse deformation.
        """
        return detjacinvphi * self.template.interpolation(
            invphi, bounds_check=False)

    def mp_deform(self, template, invphi, detjacinvphi):
        """Implementation of ``self(template, invphi, detjacinvphi)``.

        Parameters
        ----------
        template : `DiscreteLpVector`
            Fixed template deformed by the vector field. Its space
            must have the same number of dimensions as ``par_space``.
        invphi: `ProductSpaceVector`
            General inverse deformation for image grid points.
        detjacinvphi: 'DiscreteLpVector'
            The determinant of the inverse deformation.
        """
        return detjacinvphi * self.template.interpolation(
            invphi, bounds_check=False)

    def mp_euler_differential(self, invphi, detjacinvphi):
        """Eulerian differential of this operator in ``vecfield``.

        Parameters
        ----------
        invphi: `ProductSpaceVector`
            General inverse deformation for image grid points.
        detjacinvphi: 'DiscreteLpVector'
            The determinant of the inverse deformation.

        Returns
        -------
        euler_deriv_op : `Operator`
            The Eulerian differential of this operator
            evaluated at `'vecfield``.
        """
        euler_deriv_op = MPDeformationDerivative(
            self.template, invphi, detjacinvphi)
        return euler_deriv_op


class MPDeformationDerivative(MPDeformationOperator):
    """Eulerian differential of the mass-preserving deformation operator."""

    def __init__(self, template, invphi, detjacinvphi):
        """Initialize a new instance.

        Parameters
        ----------
        template : `DiscreteLpVector`
            Fixed template deformed by the vector field. Its space
            must have the same number of dimensions as ``par_space``.
        invphi: `ProductSpaceVector`
            General inverse deformation for image grid points.
        detjacinvphi: 'DiscreteLpVector'
            The determinant of the inverse deformation.
        """

        super().__init__(template)

        self.template = template
        self.invphi = invphi
        self.detjacinvphi = detjacinvphi

        Operator.__init__(self, self.invphi.space, self.template.space,
                          linear=True)

    def _call(self, vecfield):
        """Implementation of ``self(vecfield)``.

        Parameters
        ----------
        vecfield: `ProductSpaceVector`
            Like the mass preserving deformation parameters
            for image grid points. Should be in the same space
            as invphi.
        """
        mp_deform_template = self.mp_deform(
            self.template, self.invphi, self.detjacinvphi)
        tmp = self.invphi.space.element(
            [vf * mp_deform_template for vf in vecfield])
        div = odl.Divergence(self._domain)
        return -div(tmp)

    @property
    def adjoint(self):
        """Adjoint of the mass-preserving template deformation derivative."""
        adj_op = MPDeformationDerivativeAdjoint(
            self.template, self.invphi, self.detjacinvphi)
        return adj_op


class MPDeformationDerivativeAdjoint(MPDeformationOperator):
    """Adjoint of the template deformation operator derivative."""

    def __init__(self, template, invphi, detjacinvphi):
        """Initialize a new instance.

        Parameters
        ----------
        template : `DiscreteLpVector`
            Fixed template deformed by the vector field. Its space
            must have the same number of dimensions as ``par_space``.
        invphi: `ProductSpaceVector`
            General inverse deformation for image grid points.
        detjacinvphi: 'DiscreteLpVector'
            The determinant of the inverse deformation.
        """
        super().__init__(template, invphi, detjacinvphi)

        self.template = template
        self.invphi = invphi
        self.detjacinvphi = detjacinvphi

        # Switch domain and range
        self._domain, self._range = self._range, self._domain

    def _call(self, template_like):
        """Implement ``self(template_like)```.

        Parameters
        ----------
        template_like : `DiscreteLpVector`
            Element of the image space.
        """
        mp_deform_template = self.mp_deform(
            self.template, self.invphi, self.detjacinvphi)

        grad = odl.Gradient(self._domain)
        template_like_grad = grad(template_like)

        return self.invphi.space.element(
            [tlg * mp_deform_template for tlg in template_like_grad])


class L2DataMatchingFunctional(Functional):

    """Basic data matching functional using the L2 norm.

    This functional computes::

        1/2 * ||f - g||_2^2

    for a given element ``g``.
    """

    def __init__(self, space, data):
        """Initialize a new instance.

        Parameters
        ----------
        space : `DiscreteLp` with exponent 2.0
            Space where the data is matched
        data : `DiscreteLp` element-like
            Data which is to be matched
        """
        if not (isinstance(space, odl.DiscreteLp) and space.exponent == 2.0):
            raise ValueError('not an L2 space.')
        super().__init__(space, linear=False)
        self.data = self.domain.element(data)

    def _call(self, x):
        """Return ``self(x)``."""
        return self.domain.dist(x, self.data)

    def gradient(self, x):
        """Return the gradient in the point ``x``."""
        return x - self.data

    def derivative(self, x):
        """Return the derivative in ``x``."""
        return self.gradient(x).T


def generate_optimized_density_match_L2_gradient_rec(image):
    s = image.shape[0]
    if (len(image.shape) != 2):
        raise(NotImplementedError('Only 2d images are allowed so far.'))
    if (image.shape[1] != s):
        raise(NotImplementedError('Only square images are allowed so far.'))
    if (image.dtype != np.float64):
        raise(NotImplementedError('Only float64 images are allowed so far.'))

    @numba.njit('void(f8,f8[:,:],f8[:,:],f8[:,:],f8[:,:],f8[:,:],f8[:,:])')
    def density_match_L2_gradient_2d_rec(sigma, dsqrtJdx, dsqrtJdy,
                                         dtmpdx, dtmpdy, doutdx, doutdy):
        for i in xrange(s):
            for j in xrange(s):
                doutdx[i, j] = sigma*dsqrtJdx[i, j] + 2. * dtmpdx[i, j]
                doutdy[i, j] = sigma*dsqrtJdy[i, j] + 2. * dtmpdy[i, j]

    return density_match_L2_gradient_2d_rec


def proj_noise(proj_data_shape, mu=0.0, sigma=0.1):
    """Produce white Gaussian noise for projections of n-D images.

       Produce white Gaussian noise for projections, with the same size
       as the number of projections.

       Parameters
       ----------
       proj_data_shape : shape of the projections
           Give the size of noise
       mu : Mean of the norm distribution
           The defalt is 0.0
       sigma : Standard deviation of the norm distribution
           The defalt is 0.1.
    """

    return np.random.normal(mu, sigma, proj_data_shape)


def SNR(signal, noise):
    """Compute the signal-to-noise ratio in dB.
    This compute::

    SNR = 10 * log10 (
        |signal - mean(signal)| / |noise - mean(noise)|)

    Parameters
    ----------
    signal : projection
    noise : white noise
    """
    if np.abs(np.asarray(noise)).sum() != 0:
        ave1 = np.sum(signal)/signal.size
        ave2 = np.sum(noise)/noise.size
        en1 = np.sqrt(np.sum((signal - ave1) * (signal - ave1)))
        en2 = np.sqrt(np.sum((noise - ave2) * (noise - ave2)))

        return 10.0 * np.log10(en1/en2)
    else:
        return 10000.


I0name = '../ddmatch/Example3 letters/c_highres.png'
I1name = '../ddmatch/Example3 letters/i_highres.png'
# I0name = 'Example3 letters/eight.png'
# I1name = 'Example3 letters/b.png'
# I0name = 'Example3 letters/v.png'
# I1name = 'Example3 letters/j.png'
# I0name = 'Example9 letters big/V.png'
# I1name = 'Example9 letters big/J.png'
# I0name = 'Example11 skulls/handnew1.png'
# I1name = 'Example11 skulls/handnew2.png'
# I0name = 'Example8 brains/DS0002AxialSlice80.png'
# I1name = 'Example8 brains/DS0003AxialSlice80.png'

I0 = plt.imread(I0name).astype('float')
I1 = plt.imread(I1name).astype('float')

I0 = I0[::2, ::2]
I1 = I1[::2, ::2]

# Create 2-D discretization reconstruction space
# The size of the domain should be proportional to the given images
discr_space = odl.uniform_discr([-16, -16],
                                [16, 16], [128, 128],
                                dtype='float32', interp='linear')

# Create the ground truth as the given image
ground_truth = discr_space.element(I0.T)

# Create the template as the given image
template = discr_space.element(I1.T)

# Compose mass-preserving operator to template
template_mass_pre = template

# Give the number of directions
num_angles = 6

# Create the uniformly distributed directions
angle_partition = odl.uniform_partition(
    0, np.pi, num_angles, nodes_on_bdry=[(True, False)])

# Create 2-D projection domain
detector_partition = odl.uniform_partition(-24, 24, 192)

# Create 2-D parallel projection geometry
geometry = odl.tomo.Parallel2dGeometry(angle_partition,
                                       detector_partition)

# Create projection data by given setting
xray_trafo_op = odl.tomo.RayTransform(discr_space, geometry, impl='astra_cuda')

# Create projection data by given setting
proj_data = xray_trafo_op(ground_truth)

# Create white Gaussian noise
noise = 5.0 * proj_data.space.element(proj_noise(proj_data.shape))

# Output the signal-to-noise ratio
print('snr = {!r}'.format(SNR(proj_data, noise)))

# Create noisy projection data
noise_proj_data = proj_data + noise

# Do the backprojection reconstruction
backproj = xray_trafo_op.adjoint(noise_proj_data)

density_match_L2_gradient_rec = \
    generate_optimized_density_match_L2_gradient_rec(I1)

# Regularization parameter, should be nonnegtive
sigma = 1000e-1

# Step size for the gradient descent method
epsilon = 0.005

# Maximum iteration number
n_iter = 500

dm = ddmatch.TwoComponentDensityMatching(source=I1, target=I0, sigma=sigma)

# Normalize the mass of template as ground truth
W1 = I1.T * np.linalg.norm(I0, 'fro')/np.linalg.norm(I1, 'fro')

# Normalized template as an element of discretization space
template = discr_space.element(W1)

# Create the memory for energy in each iteration
E = []
kE = len(E)
E = np.hstack((E, np.zeros(n_iter)))

axis = [np.linspace(
    0, I1.shape[i], I1.shape[i], endpoint=False) for i in range(I1.ndim)]
id_map = np.meshgrid(*axis)

vec_field = [np.zeros_like(I1) for _ in range(I1.ndim)]

vx = np.zeros_like(I1)
vy = np.zeros_like(I1)

tmp = list(id_map)

tmpx = dm.idx.copy()
tmpy = dm.idy.copy()

# Test time, set starting time
start = time.clock()

for k in xrange(n_iter):

    # OUTPUT
    E[k+kE] = (sigma*(dm.sqrtJ - 1)**2).sum()

    # STEP 1: update template_mass_pre
    template_array = np.asarray(template, dtype='float64')
    template_mass_pre_array = np.asarray(template_mass_pre,
                                         dtype='float64')
    dm.image_compose(template_array, dm.phiinvx,
                     dm.phiinvy, template_mass_pre_array)

    template_mass_pre_array *= dm.J
    W = template_mass_pre_array
    template_mass_pre = discr_space.element(W)

    # STEP 2: compute the L2 gradient
    tmpx_op = xray_trafo_op(template_mass_pre)
    tmpx_op -= noise_proj_data

    E[k+kE] += np.asarray(tmpx_op**2).sum()

    tmpx_op = xray_trafo_op.adjoint(tmpx_op)
    tmpx_array = np.array(tmpx_op, dtype='float64')
    dm.image_gradient(tmpx_array, dm.dtmpdx, dm.dtmpdy)
    dm.dtmpdx *= W
    dm.dtmpdy *= W

    dm.image_gradient(dm.sqrtJ, dm.dsqrtJdx, dm.dsqrtJdy)

    # Compute the L2 gradient of the energy functional
    density_match_L2_gradient_rec(sigma, dm.dsqrtJdx, dm.dsqrtJdy,
                                  dm.dtmpdx, dm.dtmpdy,
                                  vx, vy)

    # STEP 3:
    fftx = np.fft.fftn(vx)
    ffty = np.fft.fftn(vy)
    fftx *= dm.Linv
    ffty *= dm.Linv
    vx[:] = -np.fft.ifftn(fftx).real
    vy[:] = -np.fft.ifftn(ffty).real

    # STEP 4 (v = -grad E, so to compute the inverse
    # we solve \psiinv' = -epsilon*v o \psiinv)
    np.copyto(tmpx, vx)
    tmpx *= epsilon
    np.copyto(dm.psiinvx, dm.idx)
    dm.psiinvx -= tmpx
    # Compute forward phi also (only for output purposes)
    if dm.compute_phi:
        np.copyto(dm.psix, dm.idx)
        dm.psix += tmpx

    np.copyto(tmpy, vy)
    tmpy *= epsilon
    np.copyto(dm.psiinvy, dm.idy)
    dm.psiinvy -= tmpy
    # Compute forward phi also (only for output purposes)
    if dm.compute_phi:
        np.copyto(dm.psiy, dm.idy)
        dm.psiy += tmpy

    # STEP 5
    dm.diffeo_compose(dm.phiinvx, dm.phiinvy,
                      dm.psiinvx, dm.psiinvy,
                      tmpx, tmpy)
    np.copyto(dm.phiinvx, tmpx)
    np.copyto(dm.phiinvy, tmpy)
    # Compute forward phi also (only for output purposes)
    if dm.compute_phi:
        dm.diffeo_compose(dm.phix, dm.phiy, dm.psix, dm.psiy, tmpx, tmpy)
        np.copyto(dm.phix, tmpx)
        np.copyto(dm.phiy, tmpy)

    # STEP 6
    dm.image_compose(dm.J, dm.psiinvx, dm.psiinvy, dm.sqrtJ)
    np.copyto(dm.J, dm.sqrtJ)
    dm.divergence(vx, vy, dm.divv)
    dm.divv *= -epsilon
    np.exp(dm.divv, out=dm.sqrtJ)
    dm.J *= dm.sqrtJ
    np.sqrt(dm.J, out=dm.sqrtJ)

# Test time, set end time
end = time.clock()

# Output the computational time
print(end - start)

W = W.T
dm.J = dm.J.T
dm.phiinvx, dm.phiinvy = dm.phiinvy, dm.phiinvx
backproj = np.asarray(backproj)
backproj = backproj.T

dm.template_mass_pre = discr_space.element(dm.W.T)
rec_proj_data = xray_trafo_op(dm.template_mass_pre)

plt.figure(1, figsize=(28, 28))
plt.clf()

plt.subplot(3, 3, 1)
plt.imshow(I0, cmap='bone', vmin=dm.I0.min(), vmax=I0.max())
plt.colorbar()
plt.title('Ground truth')

plt.subplot(3, 3, 2)
plt.imshow(I1, cmap='bone', vmin=dm.I1.min(), vmax=I1.max())
plt.colorbar()
plt.title('Template')

plt.subplot(3, 3, 3)
plt.imshow(backproj, cmap='bone', vmin=backproj.min(), vmax=backproj.max())
plt.colorbar()
plt.title('Backprojection')

plt.subplot(3, 3, 4)
# plt.imshow(dm.W**2, cmap='bone', vmin=dm.I0.min(), vmax=dm.I0.max())
plt.imshow(W, cmap='bone', vmin=I1.min(), vmax=I1.max())
plt.colorbar()
plt.title('Reconstructed image by {!r} directions'.format(num_angles))
# plt.title('Warped image')

jac_ax = plt.subplot(3, 3, 5)
mycmap = 'PiYG'
# mycmap = 'Spectral'
# mycmap = 'PRGn'
# mycmap = 'BrBG'
plt.imshow(dm.J, cmap=mycmap, vmin=dm.J.min(), vmax=1.+(1.-dm.J.min()))
plt.gca().set_autoscalex_on(False)
plt.gca().set_autoscaley_on(False)
# plot_warp(dm.phiinvx, dm.phiinvy, downsample=8)
jac_colorbar = plt.colorbar()
plt.title('Jacobian')

plt.subplot(3, 3, 6)
ddmatch.plot_warp(dm.phiinvx, dm.phiinvy, downsample=4)
plt.axis('equal')
warplim = [dm.phiinvx.min(), dm.phiinvx.max(),
           dm.phiinvy.min(), dm.phiinvy.max()]
warplim[0] = min(warplim[0], warplim[2])
warplim[2] = warplim[0]
warplim[1] = max(warplim[1], warplim[3])
warplim[3] = warplim[1]

plt.axis(warplim)
# plt.axis('off')
plt.gca().invert_yaxis()
plt.gca().set_aspect('equal')
plt.title('Warp')

plt.subplot(3, 3, 7)
plt.title('stepsize = {!r}, $\sigma$ = {!r}'.format(epsilon, sigma))
plt.plot(E)
plt.ylabel('Energy')
# plt.gca().axes.yaxis.set_ticklabels(['0']+['']*8)
plt.gca().axes.yaxis.set_ticklabels([])
plt.grid(True)

plt.subplot(3, 3, 8)
plt.plot(np.asarray(proj_data)[0], 'b', np.asarray(noise_proj_data)[0], 'r')
plt.title('Theta=0, blue: truth_data, red: noisy_data, SNR = 9.17dB')
plt.gca().axes.yaxis.set_ticklabels([])
plt.axis([0, 191, -17, 32])

plt.subplot(3, 3, 9)
plt.plot(np.asarray(proj_data)[0], 'b', np.asarray(rec_proj_data)[0], 'r')
plt.title('Theta=0, blue: truth_data, red: rec result')
plt.gca().axes.yaxis.set_ticklabels([])
plt.axis([0, 191, -17, 32])
