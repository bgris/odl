#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 19:06:35 2017

@author: bgris
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 18:53:28 2017

@author: bgris
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 19:56:59 2017

@author: barbara
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 12:29:18 2017

@author: barbara
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 15:47:58 2017

@author: bgris
"""

import odl
import numpy as np
from matplotlib import pylab as plt

from DeformationModulesODL.deform import Kernel
from DeformationModulesODL.deform import DeformationModuleAbstract
from DeformationModulesODL.deform import SumTranslations
from DeformationModulesODL.deform import UnconstrainedAffine
from DeformationModulesODL.deform import LocalScaling
from DeformationModulesODL.deform import LocalRotation
from DeformationModulesODL.deform import EllipseMvt
from DeformationModulesODL.deform import TemporalAttachmentModulesGeom

import scipy


def snr_fun(signal, noise, impl):
    """Compute the signal-to-noise ratio.
    Parameters
    ----------
    signal : `array-like`
        Noiseless data.
    noise : `array-like`
        Noise.
    impl : {'general', 'dB'}
        Implementation method.
        'general' means SNR = variance(signal) / variance(noise),
        'dB' means SNR = 10 * log10 (variance(signal) / variance(noise)).
    Returns
    -------
    snr : `float`
        Value of signal-to-noise ratio.
        If the power of noise is zero, then the return is 'inf',
        otherwise, the computed value.
    """
    if np.abs(np.asarray(noise)).sum() != 0:
        ave1 = np.sum(signal) / signal.size
        ave2 = np.sum(noise) / noise.size
        s_power = np.sqrt(np.sum((signal - ave1) * (signal - ave1)))
        n_power = np.sqrt(np.sum((noise - ave2) * (noise - ave2)))
        if impl == 'general':
            return s_power / n_power
        elif impl == 'dB':
            return 10.0 * np.log10(s_power / n_power)
        else:
            raise ValueError('unknown `impl` {}'.format(impl))
    else:
        return float('inf')




def compute_grid_deformation(vector_fields_list, time_step, initial_grid):
    vector_fields_list = vector_fields_list
    nb_time_points = vector_fields_list.size

    grid_points = initial_grid

    for t in range(nb_time_points):
        velocity = np.empty_like(grid_points)
        for i, vi in enumerate(vector_fields_list[t]):
            velocity[i, ...] = vi.interpolation(grid_points)
        grid_points += time_step*velocity

    return grid_points


def compute_grid_deformation_list(vector_fields_list, time_step, initial_grid):
    vector_fields_list = vector_fields_list
    nb_time_points = vector_fields_list.size
    grid_list=[]
    grid_points=initial_grid.copy()
    grid_list.append(initial_grid)

    for t in range(nb_time_points):
        velocity = np.empty_like(grid_points)
        for i, vi in enumerate(vector_fields_list[t]):
            velocity[i, ...] = vi.interpolation(grid_points)
        grid_points += time_step*velocity
        grid_list.append(grid_points.copy())

    return grid_list


# As previously but check if points of the grids are in the
# Domain and if they are not the velocity is equal to zero
def compute_grid_deformation_list_bis(vector_fields_list, time_step, initial_grid):
    vector_fields_list = vector_fields_list
    nb_time_points = vector_fields_list.size
    grid_list=[]
    grid_points=initial_grid.T.copy()
    grid_list.append(initial_grid.T)
    mini=vector_fields_list[0].space[0].min_pt
    maxi=vector_fields_list[0].space[0].max_pt
    for t in range(nb_time_points):
        print(t)
        velocity = np.zeros_like(grid_points)
        for i, vi in enumerate(vector_fields_list[t]):
            for k in range(len(initial_grid.T)):
                isindomain=1
                point=grid_points[k]
                for u in range(len(mini)):
                    if (point[u]<mini[u] or point[u]>maxi[u] ):
                        isindomain=0
                if (isindomain==1):
                    velocity[k][i] = vi.interpolation(point)

        grid_points += time_step*velocity
        grid_list.append(grid_points.copy().T)

    return grid_list


def plot_grid(grid, skip):
    for i in range(0, grid.shape[1], skip):
        plt.plot(grid[0, i, :], grid[1, i, :], 'r', linewidth=0.5)

    for i in range(0, grid.shape[2], skip):
        plt.plot(grid[0, :, i], grid[1, :, i], 'r', linewidth=0.5)



#
##%% Generate dota

#I1name = '/home/bgris/Downloads/pictures/v.png'
#I0name = '/home/bgris/Downloads/pictures/j.png'
#I0 = np.rot90(plt.imread(I0name).astype('float'), -1)
#I1 = np.rot90(plt.imread(I1name).astype('float'), -1)


space = odl.uniform_discr(
    min_pt=[-16, -16], max_pt=[16, 16], shape=[256,256],
    dtype='float32', interp='linear')

val0=-0.7
val1=0.7
val2=0.5
ellipsoids=[[1.00, .6900, .9200, 0.0000, 0.0000, 0],
            [-.98, .6624, .8740, 0.0000, -.0184, 0],
            [val0, .1100, .3100, 0.2200, 0.0000, -18],
            [val1, .1600, .4100, -.2200, 0.0000, 18],
            [val1, .2100, .2500, 0.0000, 0.3500, 0],
            [val1, .0460, .0460, 0.0000, 0.1000, 0],
            [val1, .0460, .0460, 0.0000, -.1000, 0],
            [val2, .0460, .0230, -.0800, -.6050, 0],
            [val2, .0230, .0230, 0.0000, -.6060, 0],
            [val2, .0230, .0460, 0.0600, -.6050, 0]]

template=odl.phantom.ellipsoid_phantom(space, ellipsoids)


# Change of grey scale value for the target
background= 0.2
val0=-0.7 - background
val1=0.5 - background
val2=0.6 - background
ellipsoids=[[1.00, .6900, .9200, 0.0000, 0.0000, 0],
            [-.98, .6624, .8740, 0.0000, -.0184, 0],
            [val0, .1100, .3100, 0.2200, 0.0000, -18],
            [val1, .1600, .4100, -.2200, 0.0000, 18],
            [val1, .2100, .2500, 0.0000, 0.3500, 0],
            [val1, .0460, .0460, 0.0000, 0.1000, 0],
            [val1, .0460, .0460, 0.0000, -.1000, 0],
            [val2, .0460, .0230, -.0800, -.6050, 0],
            [val2, .0230, .0230, 0.0000, -.6060, 0],
            [val2, .0230, .0460, 0.0600, -.6050, 0]]

I0=odl.phantom.ellipsoid_phantom(space, ellipsoids) + background

#template= odl.phantom.shepp_logan(space)
I0= space.element(scipy.ndimage.filters.gaussian_filter(I0.asarray(),1))
#template.show(clim=[1,1.1])


NAffine=2
kernelaff=Kernel.GaussianKernel(3)
affine=UnconstrainedAffine.UnconstrainedAffine(space, NAffine, kernelaff)

GD_affine=affine.GDspace.element([[-5,5],[3,4]])
Cont_affine=-1*affine.Contspace.element([[[0.5,0],[1,-1],[1,1]],[[-1,0.5],[-1,0],[0.5,0]]])
vect_field_affine=affine.ComputeField(GD_affine,Cont_affine)

I1=template.space.element(odl.deform.linearized._linear_deform(I0.copy(),vect_field_affine)).copy()
I2=template.space.element(odl.deform.linearized._linear_deform(I1.copy(),vect_field_affine)).copy()
#I1_0.show(clim=[1,1.1])

#I1=template.space.element((I1_0-1.03)*1.5+ 1.04)
if False:
    template.show('template')
    I1.show('I1')
    I2.show('I2')
#
##%%
#I1.show(clim=[1,1.1])

#ground_truth.show('ground truth')
#template.show('template')

# The parameter for kernel function
sigma = 3

# Give kernel function
def kernel(x):
    scaled = [xi ** 2 / (2 * sigma ** 2) for xi in x]
    return np.exp(-sum(scaled))

# Maximum iteration number
niter = 200

# Give step size for solver
eps = 0.02

# Give regularization parameter
lamb = 1*1e-15
tau = 1* 1e-2

# Give the number of directions
num_angles = 10

## Create the uniformly distributed directions
angle_partition = odl.uniform_partition(0.0, np.pi, num_angles,
                                    nodes_on_bdry=[(True, True)])
#
## Create 2-D projection domain
## The length should be 1.5 times of that of the reconstruction space
rec_space=space
detector_partition = odl.uniform_partition(-24, 24, int(round(rec_space.shape[0]*np.sqrt(2))))

# Create 2-D parallel projection geometry
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

# Ray transform aka forward projection. We use ASTRA CUDA backend.
forward_op = odl.tomo.RayTransform(rec_space, geometry, impl='astra_cpu')

# Create projection data by calling the op on the phantom
ground_truth=[rec_space.element(I1)]
proj_data = [forward_op(ground_truth[u]) for u in range(len(ground_truth))]

# Add white Gaussion noise onto the noiseless data
noise =0.25 * odl.phantom.noise.white_noise(forward_op.range)

# Create the noisy projection data
#noise_proj_data = proj_data + noise

# Give the number of time points
time_itvs = 20
nb_time_point_int=time_itvs

import copy
#data=[noise_proj_data]
data=copy.deepcopy(proj_data)
data_time_points=np.array([1])
forward_operators=[forward_op, forward_op]
Norm=odl.solvers.L2NormSquared(forward_op.range)


if False:
    np.savetxt('/home/barbara/odl/examples/Metamorphosis/SheppLogan/SheppLoganModifiedSource',template)
    np.savetxt('/home/barbara/odl/examples/Metamorphosis/SheppLogan/SheppLoganModifiedTarget',data[0])

    templateload=space.element(np.loadtxt('/home/barbara/odl/examples/Metamorphosis/SheppLogan/SheppLoganModifiedSource'))
#
#lam_fbp=0.5
#fbp = odl.tomo.fbp_op(forward_op, filter_type='Hann', frequency_scaling=lam_fbp)
#reco_fbp=fbp(data[0])
##reco_fbp.show()
#reco_fbp.show(clim=[-0.2,1.2])
##%% Define energy operator

functional=odl.deform.TemporalAttachmentMetamorphosisGeom(nb_time_point_int,
                            lamb,tau,template ,data,
                            data_time_points, forward_operators,Norm, kernel,
                            domain=None)



energy_op=odl.deform.TemporalAttachmentLDDMMGeom(nb_time_point_int, template ,data,
                            data_time_points, forward_operators,Norm, kernel,
                            domain=None)


##%% Gradient descent initialisatio
niter=200
epsV=0.002
epsZ=0.00002
X_init=functional.domain.zero()
X=X_init.copy()
##%%Gradient descent
energy=functional(X)
print(" Initial ,  energy : {}".format(energy))

for k in range(niter):
    #X[1][0].show("Iter = {}".format(k),clim=[-1,1])
    #grad=functional.gradient(X)
    grad=functional.gradient_gradzeta_penalized(X)
    X_temp0=X.copy()
    X_temp0[0]= (X[0]- epsV *grad[0]).copy()
    X_temp0[1]= (X[1]- epsZ *grad[1]).copy()
    energy_temp0=functional(X_temp0)
    if energy_temp0<energy:
        X=X_temp0.copy()
        energy=energy_temp0
        epsV*=1.1
        epsZ*=1.1
        print(" iter : {}  , energy : {}, epsV = {} , epsZ = {}".format(k,energy,epsV, epsZ))
    else:
        X_temp1=X.copy()
        X_temp1[0]= (X[0]- epsV *grad[0]).copy()
        X_temp1[1]= (X[1]- 0.5*epsZ *grad[1]).copy()
        energy_temp1=functional(X_temp1)

        X_temp2=X.copy()
        X_temp2[0]= (X[0]- 0.5*epsV *grad[0]).copy()
        X_temp2[1]= (X[1]- epsZ *grad[1]).copy()
        energy_temp2=functional(X_temp2)

        X_temp3=X.copy()
        X_temp3[0]= (X[0]- 0.5*epsV *grad[0]).copy()
        X_temp3[1]= (X[1]- 0.5*epsZ *grad[1]).copy()
        energy_temp3=functional(X_temp3)

        if (energy_temp3<=energy_temp1 and energy_temp3<=energy_temp2):
            X_temp0=X_temp3.copy()
            energy_temp0=energy_temp3
            epsZ*=0.5
            epsV*=0.5
        else:
            if (energy_temp1<=energy_temp2 and energy_temp1<=energy_temp3):
                X_temp0=X_temp1.copy()
                energy_temp0=energy_temp1
                epsZ*=0.5
            else:
                X_temp0=X_temp2.copy()
                energy_temp0=energy_temp2
                epsV*=0.5

        if energy_temp0<energy:
            X=X_temp0.copy()
            energy=energy_temp0
            print(" iter : {}  , energy : {}, epsV = {} , epsZ = {}".format(k,energy,epsV, epsZ))
        else:
            print("epsV = {} , epsZ = {}".format(epsV, epsZ))



#

##%%
mini=-1
maxi=1
#

#nameinit='/home/barbara/Results/Metamorphosis/SheppLogan/Shepp_Logan_modifie_background_0_2_target2_no_noise'
#name0= nameinit + 'nb_int_20_Metamorphosis_direct_sigma_3_lam_1_e__15_tau_1_e__2_iter_100_data_time_1__0_5'
#name0= nameinit + '_zetagradientpenalized'


nameinit='/home/bgris/Results/Metamorphosis/SheppLogan/Shepp_Logan_modified_background_0_2_target1_no_noise'
name0= nameinit + 'nb_int_20_Metamorphosis_nbangle_10_sigma_3_lam_1_e__15_tau_1_e__2_iter_100_data_time_1'
name0+= '_zetagradientpenalized'


#
#name0= nameinit + 'LDDMM_nbangle_10_sigma_5_lam_1_e__15_iter_200'


##%%
#lam_fbp=200
#fbp = odl.tomo.fbp_op(forward_op, filter_type='Hann', frequency_scaling=lam_fbp)
#reco_fbp=fbp(data[0])
##reco_fbp.show()
##reco_fbp.show(clim=[-0.2,1.2])
#
#plt.figure(1, figsize=(24, 24))
#
#plt.imshow(np.rot90(reco_fbp), cmap='bone',
#           vmin=mini,
#           vmax=maxi)
#plt.colorbar()
#namefbp=nameinit + 'fbp.png'
#plt.savefig(namefbp, bbox_inches='tight')
##%% Compute estimated trajectory
image_list_data=functional.ComputeMetamorphosis(X[0],X[1])

#image_list_data[0].show()
#image_list_data[0].show(clim=[0,1])

image_list=functional.ComputeMetamorphosisListInt(X[0],X[1])

#for i in range(nb_time_point_int+1):
#    #image_list[i].show('Metamorphosis time {}'.format(i))
#    #image_list[i].show('Metamorphosis time {}'.format(i),clim=[0,1])
#    image_list[i].show('Metamorphosis time {}'.format(i),clim=[-0.2,1.2])


template_evo=odl.deform.ShootTemplateFromVectorFields(X[0], template)

#for i in range(nb_time_point_int+1):
#    template_evo[i].show('Template evolution time {} '.format(i))
#

zeta_transp=odl.deform.ShootSourceTermBackwardlist(X[0],X[1])
#template_evolution=odl.deform.IntegrateTemplateEvol(functional.template,zeta_transp,0,functional.N)

image_evol=odl.deform.IntegrateTemplateEvol(template,zeta_transp,0,nb_time_point_int)
#for i in range(nb_time_point_int+1):
#    image_evol[i].show('Image evolution time {} '.format(i),clim=[1,1.1])


#
#grid_points=compute_grid_deformation_list(X[0], 1/nb_time_point_int, template.space.points().T)

#
#for t in range(nb_time_point_int):
#    grid=grid_points[t].reshape(2, 128, 128).copy()
#plot_grid(grid, 2)
##%%
#image_list=odl.ProductSpace(functional.template.space,functional.N).element()
#zeta_transp=odl.deform.ShootSourceTermBackwardlist(X[0], X[1]).copy()
#template_evolution=odl.deform.IntegrateTemplateEvol(functional.template,zeta_transp,0,functional.N)
#odl.deform.ShootTemplateFromVectorFieldsFinal(X[0],template_evolution[k],0,k).copy()

##%% Plot metamorphosis
image_N0= image_list
rec_result_1 = rec_space.element(image_N0[time_itvs // 4])
rec_result_2 = rec_space.element(image_N0[time_itvs // 4 * 2])
rec_result_3 = rec_space.element(image_N0[time_itvs // 4 * 3])
rec_result = rec_space.element(image_N0[time_itvs])

# Compute the projections of the reconstructed image
rec_proj_data = forward_op(rec_result)
##%%
#
#for t in range(nb_time_point_int+1):
#    #grid=grid_points[t].reshape(2, rec_space.shape[0], rec_space.shape[1]).copy()
#    image_N0[t].show('t= {}'.format(t))
#    #plot_grid(grid, 2)
##
##%%%
# Plot the results of interest
plt.figure(2, figsize=(24, 24))
#plt.clf()

plt.subplot(3, 3, 1)
plt.imshow(np.rot90(template), cmap='bone',
           vmin=mini,
           vmax=maxi)
plt.axis('off')
#plt.savefig("/home/chchen/SwedenWork_Chong/NumericalResults_S/LDDMM_results/J_V/template_J.png", bbox_inches='tight')
plt.colorbar()
plt.title('Metamorphosis')

plt.subplot(3, 3, 2)
plt.imshow(np.rot90(rec_result_1), cmap='bone',
           vmin=mini,
           vmax=maxi)

plt.axis('off')
plt.colorbar()
plt.title('time_pts = {!r}'.format(time_itvs // 4))

plt.subplot(3, 3, 3)
plt.imshow(np.rot90(rec_result_2), cmap='bone',
           vmin=mini,
           vmax=maxi)
#grid=grid_points[time_itvs // 4].reshape(2, rec_space.shape[0], rec_space.shape[1]).copy()
#plot_grid(grid, 2)
plt.axis('off')
plt.colorbar()
plt.title('time_pts = {!r}'.format(time_itvs // 4 * 2))

plt.subplot(3, 3, 4)
plt.imshow(np.rot90(rec_result_3), cmap='bone',
           vmin=mini,
           vmax=maxi)
#grid=grid_points[time_itvs // 4*2].reshape(2, rec_space.shape[0], rec_space.shape[1]).copy()
#plot_grid(grid, 2)
plt.axis('off')
plt.colorbar()
plt.title('time_pts = {!r}'.format(time_itvs // 4 * 3))

plt.subplot(3, 3, 5)
plt.imshow(np.rot90(rec_result), cmap='bone',
           vmin=mini,
           vmax=maxi)
#grid=grid_points[time_itvs // 4*3].reshape(2, rec_space.shape[0], rec_space.shape[1]).copy()
#plot_grid(grid, 2)
plt.axis('off')
plt.colorbar()
#plt.title('Reconstructed by {!r} iters, '
#    '{!r} projs'.format(niter, num_angles))

plt.subplot(3, 3, 7)
plt.imshow(np.rot90(ground_truth[0]), cmap='bone',
           vmin=mini,
           vmax=maxi)
plt.axis('off')
plt.colorbar()
plt.title('Ground truth ')



name=name0 + 'metamorphosis.png'
plt.savefig(name, bbox_inches='tight')

##%% Plot template
image_N0= template_evo
rec_result_1 = rec_space.element(image_N0[time_itvs // 4])
rec_result_2 = rec_space.element(image_N0[time_itvs // 4 * 2])
rec_result_3 = rec_space.element(image_N0[time_itvs // 4 * 3])
rec_result = rec_space.element(image_N0[time_itvs])
##%%
#
#for t in range(nb_time_point_int+1):
#    #grid=grid_points[t].reshape(2, rec_space.shape[0], rec_space.shape[1]).copy()
#    image_N0[t].show('t= {}'.format(t))
#    #plot_grid(grid, 2)
##
##%%%
# Plot the results of interest
plt.figure(3, figsize=(24, 24))
#plt.clf()

plt.subplot(3, 3, 1)
plt.imshow(np.rot90(template), cmap='bone',
           vmin=mini,
           vmax=maxi)
plt.axis('off')
#plt.savefig("/home/chchen/SwedenWork_Chong/NumericalResults_S/LDDMM_results/J_V/template_J.png", bbox_inches='tight')
plt.colorbar()
plt.title('Template')

plt.subplot(3, 3, 2)
plt.imshow(np.rot90(rec_result_1), cmap='bone',
           vmin=mini,
           vmax=maxi)

plt.axis('off')
plt.colorbar()
plt.title('time_pts = {!r}'.format(time_itvs // 4))

plt.subplot(3, 3, 3)
plt.imshow(np.rot90(rec_result_2), cmap='bone',
           vmin=mini,
           vmax=maxi)
#grid=grid_points[time_itvs // 4].reshape(2, rec_space.shape[0], rec_space.shape[1]).copy()
#plot_grid(grid, 2)
plt.axis('off')
plt.colorbar()
plt.title('time_pts = {!r}'.format(time_itvs // 4 * 2))

plt.subplot(3, 3, 4)
plt.imshow(np.rot90(rec_result_3), cmap='bone',
           vmin=mini,
           vmax=maxi)
#grid=grid_points[time_itvs // 4*2].reshape(2, rec_space.shape[0], rec_space.shape[1]).copy()
#plot_grid(grid, 2)
plt.axis('off')
plt.colorbar()
plt.title('time_pts = {!r}'.format(time_itvs // 4 * 3))

plt.subplot(3, 3, 5)
plt.imshow(np.rot90(rec_result), cmap='bone',
           vmin=mini,
           vmax=maxi)
#grid=grid_points[time_itvs // 4*3].reshape(2, rec_space.shape[0], rec_space.shape[1]).copy()
#plot_grid(grid, 2)
#plt.axis('off')
#plt.colorbar()
#plt.title('Reconstructed by {!r} iters, '
#    '{!r} projs'.format(niter, num_angles))

plt.subplot(3, 3, 7)
plt.imshow(np.rot90(ground_truth[0]), cmap='bone',
           vmin=mini,
           vmax=maxi)
plt.axis('off')
plt.colorbar()
plt.title('Ground truth ')




name=name0 + 'template.png'
plt.savefig(name, bbox_inches='tight')

##%% Plot image
image_N0= image_evol
rec_result_1 = rec_space.element(image_N0[time_itvs // 4])
rec_result_2 = rec_space.element(image_N0[time_itvs // 4 * 2])
rec_result_3 = rec_space.element(image_N0[time_itvs // 4 * 3])
rec_result = rec_space.element(image_N0[time_itvs])

##%%
#
#for t in range(nb_time_point_int+1):
#    #grid=grid_points[t].reshape(2, rec_space.shape[0], rec_space.shape[1]).copy()
#    image_N0[t].show('t= {}'.format(t))
#    #plot_grid(grid, 2)
##
##%%%
# Plot the results of interest
plt.figure(4, figsize=(24, 24))
#plt.clf()

plt.subplot(3, 3, 1)
plt.imshow(np.rot90(template), cmap='bone',
           vmin=mini,
           vmax=maxi)
plt.axis('off')
#plt.savefig("/home/chchen/SwedenWork_Chong/NumericalResults_S/LDDMM_results/J_V/template_J.png", bbox_inches='tight')
plt.colorbar()
plt.title('Image')

plt.subplot(3, 3, 2)
plt.imshow(np.rot90(rec_result_1), cmap='bone',
           vmin=mini,
           vmax=maxi)

plt.axis('off')
plt.colorbar()
plt.title('time_pts = {!r}'.format(time_itvs // 4))

plt.subplot(3, 3, 3)
plt.imshow(np.rot90(rec_result_2), cmap='bone',
           vmin=mini,
           vmax=maxi)
#grid=grid_points[time_itvs // 4].reshape(2, rec_space.shape[0], rec_space.shape[1]).copy()
#plot_grid(grid, 2)
plt.axis('off')
plt.colorbar()
plt.title('time_pts = {!r}'.format(time_itvs // 4 * 2))

plt.subplot(3, 3, 4)
plt.imshow(np.rot90(rec_result_3), cmap='bone',
           vmin=mini,
           vmax=maxi)
#grid=grid_points[time_itvs // 4*2].reshape(2, rec_space.shape[0], rec_space.shape[1]).copy()
#plot_grid(grid, 2)
plt.axis('off')
plt.colorbar()
plt.title('time_pts = {!r}'.format(time_itvs // 4 * 3))

plt.subplot(3, 3, 5)
plt.imshow(np.rot90(rec_result), cmap='bone',
           vmin=mini,
           vmax=maxi)
#grid=grid_points[time_itvs // 4*3].reshape(2, rec_space.shape[0], rec_space.shape[1]).copy()
#plot_grid(grid, 2)
#plt.axis('off')
#plt.colorbar()
#plt.title('Reconstructed by {!r} iters, '
#    '{!r} projs'.format(niter, num_angles))

plt.subplot(3, 3, 7)
plt.imshow(np.rot90(ground_truth[0]), cmap='bone',
           vmin=mini,
           vmax=maxi)
plt.axis('off')
plt.colorbar()
plt.title('Ground truth')


#    plt.title('$\Theta=90^\circ$')
#    plt.gca().axes.yaxis.set_ticklabels([])


name=name0 + 'image.png'
plt.savefig(name, bbox_inches='tight')
