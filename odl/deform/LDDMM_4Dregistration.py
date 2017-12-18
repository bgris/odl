#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 14:07:07 2017

@author: bgris
"""

"""Operators and functions for 4D image registration via LDDMM."""

# Imports for common Python 2/3 codebase
#from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

import numpy as np

from odl.discr import DiscreteLp, Gradient, Divergence
from odl.operator import Operator, PointwiseInner
from odl.space import ProductSpace
from odl.deform.linearized import _linear_deform
from odl.discr import (uniform_discr, ResizingOperator)
from odl.operator import (DiagonalOperator, IdentityOperator)
from odl.trafos import FourierTransform
from odl.space.fspace import FunctionSpace
import odl

from odl.solvers.functional.functional import Functional
__all__ = ('TemporalAttachmentLDDMMGeom', 'RegularityLDDMM',
           'ShootTemplateFromVectorFields','RegularityGrowth' )


def padded_ft_op(space, padded_size):
    """Create zero-padding fft setting

    Parameters
    ----------
    space : the space needs to do FT
    padding_size : the percent for zero padding
    """
    padded_op = ResizingOperator(
        space, ran_shp=[padded_size for _ in range(space.ndim)])
    shifts = [not s % 2 for s in space.shape]
    ft_op = FourierTransform(
        padded_op.range, halfcomplex=False, shift=shifts)

    return ft_op * padded_op

def fitting_kernel(space, kernel):

    kspace = ProductSpace(space, space.ndim)

    # Create the array of kernel values on the grid points
    discretized_kernel = kspace.element(
        [space.element(kernel) for _ in range(space.ndim)])
    return discretized_kernel

def ShootTemplateFromVectorFields(vector_field_list, template):
    N=vector_field_list.size-1
    series_image_space_integration = ProductSpace(template.space,
                                                  N+1)
    inv_N=1/N
    I=series_image_space_integration.element()
    I[0]=template.copy()
    for i in range(N):
        I[i+1]=template.space.element(
                _linear_deform(I[i],
                               -inv_N * vector_field_list[i])).copy()
    return I

class TemporalAttachmentLDDMMGeom(Functional):

    """Deformation operator with fixed template acting on displacement fields.



        Compute the attachment term of the deformed template, copared to the
        given data at each time point.


    """


    def __init__(self, nb_time_point_int, template, data, data_time_points, forward_operators,Norm, kernel, domain=None):
        """
        Parameters
        ----------
        nb_time_point_int : int
           number of time points for the numerical integration
        forward_operators : list of `Operator`
            list of the forward operators, one per data time point.
        Norm : functional
            Norm in the data space (ex: l2 norm)
        data_elem : list of 'DiscreteLpElement'
            Given data.
        data_time_points : list of floats
            time of the data (between 0 and 1)
        I : `DiscreteLpElement`
            Fixed template deformed by the deformation.
        kernel : 'function'
            Kernel function in RKHS.

        """

        self.template=template
        self.data=data
        self.Norm=Norm
        self.data_time_points= np.array(data_time_points)
        self.forward_op=forward_operators
        self.kernel=kernel
        self.nb_data=self.data_time_points.size
        self.image_domain=forward_operators[0].domain
        # Give the number of time intervals
        self.N = nb_time_point_int

        # Give the inverse of time intervals for the integration
        self.inv_N = 1.0 / self.N # the indexes will go from 0 (t=0) to N (t=1)

        # list of indexes k_j such that
        # k_j /N <= data_time_points[j] < (k_j +1) /N
        self.k_j_list=np.arange(self.nb_data)
        for j in range(self.nb_data):
            self.k_j_list[j]= int(self.N*data_time_points[j])

        # list of indexes j_k such that
        #  data_time_points[j_k] < k/N <= data_time_points[j_k +1]
        self.j_k_list=np.arange(self.N+1)
        for k in range(self.N+1):
            for j in range(self.nb_data):
                if data_time_points[self.k_j_list.size -1-j]>=k/self.N:
                    self.j_k_list[k]= int(self.k_j_list.size -1-j)


        # Definition S[j] = Norm(forward_operators[j] - data[j])
        S=[]
        for j in range(self.nb_data):
            S.append(self.Norm*(self.forward_op[j] - self.data[j]))
        self.S=S


        # the domain is lists of vector fields
        if domain is None:
            domain = odl.ProductSpace(
                    self.template.space.real_space.tangent_bundle,self.N+1)


        super().__init__(domain,linear=False)


    def _call(self, vector_field_list, out=None):

        series_image_space_integration = ProductSpace(self.image_domain, self.N+1)
        series_image_space_data = ProductSpace(self.image_domain, self.nb_data)
        image_integration = series_image_space_integration.element()
        image_integration[0] = self.image_domain.element(self.template).copy()
        image_data = series_image_space_data.element()
        # Integration of the transport of the template
        for i in range(self.N):
            # Update image_N0[i+1] by image_N0[i] and vector_fields[i+1]
            image_integration[i+1] = self.image_domain.element(
                _linear_deform(image_integration[i],
                               -self.inv_N * vector_field_list[i])).copy()

        # time interpolation to obtain the transporter template at the
        # data time points : linear deformation from
        # image_integration[self.k_j_list[j]] with
        #  delta0* vector_field_list[self.k_j_list[j]]
        for j in range(self.nb_data):
                delta0=(self.data_time_points[j] -((self.k_j_list[j])/self.N))
                image_data[j]=self.image_domain.element(
                        _linear_deform(image_integration[self.k_j_list[j]],
                        -delta0 * vector_field_list[self.k_j_list[j]])).copy()

        energy=0
        for j in range(self.nb_data):
            #energy+=self.Norm(self.forward_op[j](image_data[j]) -  self.data[j])
            energy+=self.S[j](image_data[j])

        return energy


    @property
    def gradient(self):

        functional = self

        class TemporalAttachmentLDDMMGeomGradient(Operator):

            """The gradient operator of the TemporalAttachmentLDDMMGeom
            functional."""

            def __init__(self):
                """Initialize a new instance."""
                super().__init__(functional.domain, functional.domain,
                                 linear=False)

            def _call(self, vector_field_list):

                # Get the dimension
                dim = functional.image_domain.ndim

                # Compute the transport of the template
                series_image_space_integration = ProductSpace(functional.image_domain, functional.N+1)
                series_image_space_data = ProductSpace(functional.image_domain, functional.nb_data)
                image_integration = series_image_space_integration.element()
                image_integration[0] = functional.image_domain.element(functional.template).copy()
                image_data = series_image_space_data.element()
                vector_field_list_data=odl.ProductSpace(
                  functional.image_domain.tangent_bundle,functional.nb_data).element()

                # Integration of the transport of the template
                for i in range(functional.N):
                    # Update image_N0[i+1] by image_N0[i] and vector_fields[i+1]
                    image_integration[i+1] = functional.image_domain.element(
                        _linear_deform(image_integration[i],
                                       -functional.inv_N * vector_field_list[i])).copy()

                # time interpolation to obtain at the data time points
                # - the transported template  : linear deformation from
                #    image_integration[self.k_j_list[j]] with
                #    delta0* vector_field_list[self.k_j_list[j]]
                # - the vector field  : interpolation
                for j in range(functional.nb_data):
                        delta0=(functional.data_time_points[j] -((functional.k_j_list[j])/functional.N))
                        image_data[j]=functional.image_domain.element(
                                _linear_deform(image_integration[functional.k_j_list[j]],
                                -delta0 * vector_field_list[functional.k_j_list[j]])).copy()
                        delta1=1-functional.N*delta0
                        v0=vector_field_list[functional.k_j_list[j]]
                        if functional.k_j_list[j]<functional.N:
                            v1=vector_field_list[functional.k_j_list[j]+1]
                            vector_field_list_data[j]=delta0*v1+delta1*v0
                        else:
                            vector_field_list_data[j]=v0

                # FFT setting for data matching term, 1 means 100% padding
                padded_size = 2 * functional.image_domain.shape[0]
                padded_ft_fit_op = padded_ft_op(functional.image_domain, padded_size)
                vectorial_ft_fit_op = DiagonalOperator(*([padded_ft_fit_op] * dim))

                # Compute the FT of kernel in fitting term
                discretized_kernel = fitting_kernel(functional.image_domain, functional.kernel)
                ft_kernel_fitting = vectorial_ft_fit_op(discretized_kernel)

                # Computation of h_tj_tauk and then the gradient
                grad=functional.domain.zero()

                # Create the gradient op
                grad_op = Gradient(domain=functional.image_domain, method='forward',
                       pad_mode='symmetric')
                # Create the divergence op
                div_op = -grad_op.adjoint
#                detDphi_N1 = series_image_space_integration.element()

                for j in range(functional.nb_data):
                    grad_S=series_image_space_integration.element()
                    detDphi=series_image_space_integration.element()
                    # initialization at time t_j
                    #detDphi_t_j=image_domain.one() # initialization not necessary here
                    grad_S_tj=functional.S[j].gradient(image_data[j]).copy()

#                    grad_S_tj=((functional.forward_op[j].adjoint * (functional.forward_op[j] - functional.data[j]))(image_data[j])).copy()
                    # computation at time tau_k_j
                    delta_t= functional.data_time_points[j]-(functional.k_j_list[j]*functional.inv_N)
                    detDphi[functional.k_j_list[j]]=functional.image_domain.element(
                                      1+delta_t *
                                      div_op(vector_field_list_data[j])).copy()
                    grad_S[functional.k_j_list[j]]=functional.image_domain.element(
                                   _linear_deform(grad_S_tj,
                                   delta_t * vector_field_list_data[j])).copy()


                    # Because it influences the gradient only if k_j_list[j] < t_j (not if it is equal)
                    if (not delta_t==0):
                        tmp=grad_op(image_integration[functional.k_j_list[j]]).copy()
                        tmp1=(grad_S[functional.k_j_list[j]] *detDphi[functional.k_j_list[j]]).copy()
                        for d in range(dim):
                            tmp[d] *= tmp1
                        tmp3= (2 * np.pi) ** (dim / 2.0) * vectorial_ft_fit_op.inverse(vectorial_ft_fit_op(tmp) * ft_kernel_fitting)
                        grad[functional.k_j_list[j]]-=tmp3.copy()
                    # loop for k < k_j
                    delta_t= functional.inv_N
                    for u in range(functional.k_j_list[j]):
                        k=functional.k_j_list[j] -u-1
                        detDphi[k]=functional.image_domain.element(
                                _linear_deform(detDphi[k+1],
                                delta_t*vector_field_list[k])).copy()
                        detDphi[k]=functional.image_domain.element(detDphi[k]*
                                   functional.image_domain.element(1+delta_t *
                                     div_op(vector_field_list[k]))).copy()
                        grad_S[k]=functional.image_domain.element(
                                       _linear_deform(grad_S[k+1],
                                       delta_t * vector_field_list[k])).copy()

                        tmp= grad_op(image_integration[k]).copy()
                        tmp1=(grad_S[k] * detDphi[k]).copy()
                        for d in range(dim):
                            tmp[d] *= tmp1

                        tmp3= (2 * np.pi) ** (dim / 2.0) * vectorial_ft_fit_op.inverse(
                        vectorial_ft_fit_op(tmp) * ft_kernel_fitting)

                        grad[k]-=tmp3.copy()

                return grad

        return TemporalAttachmentLDDMMGeomGradient()


    @property
    def gradientL2(self):

        functional = self

        class TemporalAttachmentLDDMMGeomGradient(Operator):

            """The gradient operator of the TemporalAttachmentLDDMMGeom
            functional."""

            def __init__(self):
                """Initialize a new instance."""
                super().__init__(functional.domain, functional.domain,
                                 linear=False)

            def _call(self, vector_field_list):

                # Get the dimension
                dim = functional.image_domain.ndim

                # Compute the transport of the template
                series_image_space_integration = ProductSpace(functional.image_domain, functional.N+1)
                series_image_space_data = ProductSpace(functional.image_domain, functional.nb_data)
                image_integration = series_image_space_integration.element()
                image_integration[0] = functional.image_domain.element(functional.template).copy()
                image_data = series_image_space_data.element()
                vector_field_list_data=odl.ProductSpace(
                  functional.image_domain.tangent_bundle,functional.nb_data).element()

                # Integration of the transport of the template
                for i in range(functional.N):
                    # Update image_N0[i+1] by image_N0[i] and vector_fields[i+1]
                    image_integration[i+1] = functional.image_domain.element(
                        _linear_deform(image_integration[i],
                                       -functional.inv_N * vector_field_list[i])).copy()

                # time interpolation to obtain at the data time points
                # - the transported template  : linear deformation from
                #    image_integration[self.k_j_list[j]] with
                #    delta0* vector_field_list[self.k_j_list[j]]
                # - the vector field  : interpolation
                for j in range(functional.nb_data):
                        delta0=(functional.data_time_points[j] -((functional.k_j_list[j])/functional.N))
                        image_data[j]=functional.image_domain.element(
                                _linear_deform(image_integration[functional.k_j_list[j]],
                                -delta0 * vector_field_list[functional.k_j_list[j]])).copy()
                        delta1=1-functional.N*delta0
                        v0=vector_field_list[functional.k_j_list[j]]
                        if functional.k_j_list[j]<functional.N:
                            v1=vector_field_list[functional.k_j_list[j]+1]
                            vector_field_list_data[j]=delta0*v1+delta1*v0
                        else:
                            vector_field_list_data[j]=v0

                # FFT setting for data matching term, 1 means 100% padding
                padded_size = 2 * functional.image_domain.shape[0]
                padded_ft_fit_op = padded_ft_op(functional.image_domain, padded_size)
                vectorial_ft_fit_op = DiagonalOperator(*([padded_ft_fit_op] * dim))

                # Compute the FT of kernel in fitting term
                discretized_kernel = fitting_kernel(functional.image_domain, functional.kernel)
                ft_kernel_fitting = vectorial_ft_fit_op(discretized_kernel)

                # Computation of h_tj_tauk and then the gradient
                grad=functional.domain.zero()

                # Create the gradient op
                grad_op = Gradient(domain=functional.image_domain, method='forward',
                       pad_mode='symmetric')
                # Create the divergence op
                div_op = -grad_op.adjoint
#                detDphi_N1 = series_image_space_integration.element()

                for j in range(functional.nb_data):
                    grad_S=series_image_space_integration.element()
                    detDphi=series_image_space_integration.element()
                    # initialization at time t_j
                    #detDphi_t_j=image_domain.one() # initialization not necessary here
                    grad_S_tj=functional.S[j].gradient(image_data[j]).copy()

#                    grad_S_tj=((functional.forward_op[j].adjoint * (functional.forward_op[j] - functional.data[j]))(image_data[j])).copy()
                    # computation at time tau_k_j
                    delta_t= functional.data_time_points[j]-(functional.k_j_list[j]*functional.inv_N)
                    detDphi[functional.k_j_list[j]]=functional.image_domain.element(
                                      np.exp(delta_t *
                                      div_op(vector_field_list_data[j]))).copy()
                    grad_S[functional.k_j_list[j]]=functional.image_domain.element(
                                   _linear_deform(grad_S_tj,
                                   delta_t * vector_field_list_data[j])).copy()

                    # Because it influences the gradient only if k_j_list[j] < t_j (not if it is equal)
                    if (not delta_t==0):
                        tmp= grad_op(image_integration[functional.k_j_list[j]]).copy()
                        tmp1=(grad_S[functional.k_j_list[j]] *
                              detDphi[functional.k_j_list[j]]).copy()
                        for d in range(dim):
                            tmp[d] *= tmp1

                        grad[functional.k_j_list[j]]-=tmp.copy()

                    # loop for k < k_j
                    delta_t= functional.inv_N
                    for u in range(functional.k_j_list[j]):
                        k=functional.k_j_list[j] -u-1
                        detDphi[k]=functional.image_domain.element(
                                _linear_deform(detDphi[k+1],
                                delta_t*vector_field_list[k])).copy()
                        detDphi[k]=functional.image_domain.element(detDphi[k]*
                                   functional.image_domain.element(1+delta_t *
                                     div_op(vector_field_list[k]))).copy()
                        grad_S[k]=functional.image_domain.element(
                                       _linear_deform(grad_S[k+1],
                                       delta_t * vector_field_list[k])).copy()

                        tmp= grad_op(image_integration[k]).copy()
                        tmp1=(grad_S[k] * detDphi[k]).copy()
                        for d in range(dim):
                            tmp[d] *= tmp1


                        grad[k]-=tmp.copy()

                return grad

        return TemporalAttachmentLDDMMGeomGradient()





class RegularityLDDMM(Functional):



    def __init__(self,  kernel, domain):
        """
        Parameters
        ----------
        kernel : 'function'
            Kernel function in RKHS.
        domain : Product space
            space of the discretized time-dependent vector fields

        """

        super().__init__(domain,linear=False)



    @property
    def gradient(self):
        """Gradient operator of the Rosenbrock functional."""
        functional = self

        class RegularityLDDMMGradient(Operator):

            """The gradient operator of the RegularityLDDMM
            functional."""

            def __init__(self):
                """Initialize a new instance."""
                super().__init__(functional.domain, functional.domain,
                                 linear=False)

            def _call(self, vector_field_list):
                return vector_field_list.copy()

        return RegularityLDDMMGradient()




class RegularityGrowth(Functional):



    def __init__(self,  kernel, domain):
        """
        Parameters
        ----------
        kernel : 'function'
            Kernel function in RKHS.
        domain : Product space
            space of the discretized time-dependent vector fields

        """

        super().__init__(domain,linear=False)



    @property
    def gradient(self):

        functional = self

        class RegularityGrowthGradient(Operator):

            """The gradient operator of the RegularityLDDMM
            functional."""

            def __init__(self):
                """Initialize a new instance."""
                super().__init__(functional.domain, functional.domain,
                                 linear=False)

            def _call(self, vector_field_list):
                out=vector_field_list.copy()
                N=vector_field_list.size
                for i in range(N):
                    out[i]*=(N-i)
                return out

        return RegularityGrowthGradient()








class RegularitySobolev(Functional):



    def __init__(self,  domain, space, mu, lam, nb_time_point_int):
        
        """
        Parameters
        ----------
        domain : Product space
            space of the discretized time-dependent vector fields

        """
        self.N = nb_time_point_int
        self.mu = mu
        self.lam = lam
        self.image_space = space
        super().__init__(domain,linear=False)



    @property
    def gradient(self):
        """Gradient operator o."""
        functional = self

        class RegularitySobolevGradient(Operator):

            """The gradient operator of the Sobolev functional developed by 
            Sebastian Neumayer
            
            ONLY DIMENSION 2
        
            ."""

            def __init__(self):
                """Initialize a new instance."""
                super().__init__(functional.domain, functional.domain,
                                 linear=False)

            def _call(self, vector_field_list):
                grad=functional.domain.zero()
                dim = functional.space.ndim
                grad_op = Gradient(domain=functional.space, method='forward',
                                   pad_mode='symmetric')

                for t in range(self.N):
                    # Der_vectfield[i][j] = der_j v_i
                    Der_vectfield = [grad_op(v) for v in vector_field_list[t]]
                    # Der2_vect_field[i][j][k] = der_k der_j v_i
                    Der2_vect_field = [[grad_op(Der_vectfield[i][j]) for j in range(dim)] for i in range(dim)]
                    
                    grad[t][0] -= 2*(functional.mu + 0.5*functional.lam) * Der2_vect_field[0][0][0]
                    grad[t][0] -=  2*functional.mu * Der2_vect_field[1][0][1]
                    grad[t][0] -= functional.lam * Der2_vect_field[1][1][0]
                
                    grad[t][1] -= 2*(functional.mu + 0.5*functional.lam) * Der2_vect_field[1][1][1]
                    grad[t][1] -=  2*functional.mu * Der2_vect_field[0][1][0]
                    grad[t][1] -= functional.lam * Der2_vect_field[0][0][1]
                
                
                
                return copy.deepcopy(grad)

        return RegularityLDDMMGradient()

