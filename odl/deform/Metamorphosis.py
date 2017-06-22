#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 17:34:54 2017

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
__all__ = ('TemporalAttachmentMetamorphosisGeom', 'IntegrateTemplateEvol',
           'ShootTemplateFromVectorFieldsFinal' , 'ShootTemplateFromVectorFields' )


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
        padded_op.range, halfcomplex=False, shift=shifts, impl='pyfftw')

    return ft_op * padded_op

def fitting_kernel(space, kernel):

    kspace = ProductSpace(space, space.ndim)

    # Create the array of kernel values on the grid points
    discretized_kernel = kspace.element(
        [space.element(kernel) for _ in range(space.ndim)])
    return discretized_kernel

def ShootTemplateFromVectorFieldsFinal(vector_field_list, template, k0, k1):
    N=vector_field_list.size-1
    inv_N=1/N
    I=template.copy()
    for i in range(k0,k1):
        I=template.space.element(
                _linear_deform(I,
                               -inv_N * vector_field_list[i])).copy()
    return I

def ShootTemplateFromVectorFields(vector_field_list, template):
    N=vector_field_list.size-1
    inv_N=1/N
    series_image_space_integration = ProductSpace(template.space,N+1)
    I=series_image_space_integration.element()
    I[0]=template.copy()
    for i in range(0,N):
        I[i+1]=template.space.element(
                _linear_deform(I[i],
                               -inv_N * vector_field_list[i])).copy()
    return I

def IntegrateTemplateEvol(template,zeta,k0,k1):
    N=zeta.size-1
    inv_N=1/N
    series_image_space_integration = ProductSpace(template.space,N+1)
    I=series_image_space_integration.element()
    I[0]=template.copy()
    for i in range(k0,k1):
        I[i+1]=I[i]+ inv_N * zeta[i]
    return I

class TemporalAttachmentMetamorphosisGeom(Functional):

    """

    """


    def __init__(self, nb_time_point_int, lamb, tau, template, data, data_time_points, forward_operators,Norm, kernel, domain=None):
        """
        Parameters
        ----------
        nb_time_point_int : int
           number of time points for the numerical integration
        lamb : positive real number
            multiplies the norm of the velocity field in the functional
        tau : positive real number
            multiplies the norm of the template evolution in the functional
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
        self.lamb=lamb
        self.tau=tau
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

        dim=self.image_domain.ndim
        # FFT setting for data matching term, 1 means 100% padding
        padded_size = 2 * self.image_domain.shape[0]
        padded_ft_fit_op = padded_ft_op(self.image_domain, padded_size)
        vectorial_ft_fit_op = DiagonalOperator(*([padded_ft_fit_op] * dim))
        self.vectorial_ft_fit_op=vectorial_ft_fit_op
        # Compute the FT of kernel in fitting term
        discretized_kernel = fitting_kernel(self.image_domain, self.kernel)
        ft_kernel_fitting = vectorial_ft_fit_op(discretized_kernel)
        self.ft_kernel_fitting=ft_kernel_fitting

        # the domain is lists of vector fields
        if domain is None:
            domain = odl.ProductSpace(odl.ProductSpace(
                    self.template.space.real_space.tangent_bundle,self.N+1),
                    odl.ProductSpace(
                    self.image_domain,self.N+1))


        super().__init__(domain,linear=False)



    def _call(self,X , out=None):
        vector_field_list=X[0]
        zeta_list=X[1]

        image_list=self.ComputeMetamorphosis(vector_field_list,zeta_list)


        dim=self.image_domain.ndim
        energy=0
        for j in range(self.nb_data):
            #energy+=self.Norm(self.forward_op[j](image_data[j]) -  self.data[j])
            energy+=self.S[j](image_list[j])

        for i in range(self.N):
            temp=(2 * np.pi) ** (dim / 2.0) * self.vectorial_ft_fit_op.inverse(self.vectorial_ft_fit_op(vector_field_list[i]) * self.ft_kernel_fitting).copy()
            energy+=self.lamb*vector_field_list[i].inner(temp)
            energy+=self.tau*zeta_list[i].inner(zeta_list[i])

        return energy

    def ComputeMetamorphosis(self,vector_field_list,zeta_list,):
        image_list=ProductSpace(self.template.space,self.nb_data).element()

        template_evolution=IntegrateTemplateEvol(self.template,zeta_list,0,self.N)
        # We build for each j image_list[j]= template_t_j \circ \phi_{t_j}^-1
        for j in range(self.nb_data):
                delta0=(self.data_time_points[j] -((self.k_j_list[j])/self.N))
                template_t_j=self.image_domain.element(
                        _linear_deform(template_evolution[self.k_j_list[j]],
                        -delta0 * vector_field_list[self.k_j_list[j]])).copy()
                #image_t_j_k_j= template_t_j \circ \phi_{\tau_k_j}^-1
                image_t_j_k_j=ShootTemplateFromVectorFieldsFinal(
                        vector_field_list,template_t_j,0,self.k_j_list[j])
                image_list[j]=self.image_domain.element(
                        _linear_deform(image_t_j_k_j,
                        -delta0 * vector_field_list[self.k_j_list[j]])).copy()

        return image_list



    def ConvolveIntegrate(self,grad_S_init,H,j0,vector_field_list,zeta_list):
            dim = self.image_domain.ndim

            k_j=self.k_j_list[j0]
            h=odl.ProductSpace(self.image_domain.tangent_bundle,k_j+1).zero()
            eta=odl.ProductSpace(self.image_domain,k_j+1).zero()

            grad_op = Gradient(domain=self.image_domain, method='forward',
                   pad_mode='symmetric')
            # Create the divergence op
            div_op = -grad_op.adjoint
            delta0=self.data_time_points[j0] -(k_j/self.N)

            detDphi=self.image_domain.element(
                                      1+delta0 *
                                      div_op(vector_field_list[k_j])).copy()
            grad_S=self.image_domain.element(
                                   _linear_deform(grad_S_init,
                                   delta0 * vector_field_list[k_j])).copy()

            if (not delta0==0):
                tmp=H[k_j].copy()
                tmp1=(grad_S * detDphi).copy()
                for d in range(dim):
                    tmp[d] *= tmp1
                tmp3=(2 * np.pi) ** (dim / 2.0) * self.vectorial_ft_fit_op.inverse(self.vectorial_ft_fit_op(tmp) * self.ft_kernel_fitting)
                h[k_j]-=tmp3.copy()
                eta[k_j]-=detDphi*grad_S

            delta_t= self.inv_N
            for u in range(k_j):
                k=k_j -u-1
                detDphi=self.image_domain.element(
                                _linear_deform(detDphi,
                                   delta_t*vector_field_list[k])).copy()
                detDphi=self.image_domain.element(detDphi*
                                self.image_domain.element(1+delta_t *
                                div_op(vector_field_list[k]))).copy()
                grad_S=self.image_domain.element(
                                   _linear_deform(grad_S,
                                   delta_t * vector_field_list[k])).copy()

                tmp=H[k].copy()
                tmp1=(grad_S * detDphi).copy()
                for d in range(dim):
                    tmp[d] *= tmp1
                tmp3=(2 * np.pi) ** (dim / 2.0) * self.vectorial_ft_fit_op.inverse(self.vectorial_ft_fit_op(tmp) * self.ft_kernel_fitting)
                h[k]-=tmp3.copy()
                eta[k]-=detDphi*grad_S

            return [h,eta]



    @property
    def gradient(self):

        functional = self

        class TemporalAttachmentMetamorphosisGeomGradient(Operator):

            """The gradient operator of the TemporalAttachmentLDDMMGeom
            functional."""

            def __init__(self):
                """Initialize a new instance."""
                super().__init__(functional.domain, functional.domain,
                                 linear=False)

            def _call(self, X):

                vector_field_list=X[0]
                zeta_list=X[1]
                h=vector_field_list.space.zero()
                eta=zeta_list.space.zero()

                # Compute the metamorphosis at the data time points
                image_list=functional.ComputeMetamorphosis(vector_field_list,zeta_list)

                template_deformation=ShootTemplateFromVectorFields(vector_field_list,functional.template)

                # Computation of G_k=\nabla (template \circ \phi_{\tau_k}^-1) for each k
                G=odl.ProductSpace(functional.image_domain.tangent_bundle, functional.N +1).element()
                grad_op = Gradient(domain=functional.image_domain, method='forward',
                   pad_mode='symmetric')
                for k in range(functional.N +1):
                    G[k]=grad_op(template_deformation[k])


                # Computation of  \int_0^\tau_k \nabla (\zeta(\tau) \circ \phi_{\tau_k , \tau}) d\tau
                for k in range(functional.N +1):
                    G[k]+= (functional.inv_N * grad_op(zeta_list[k])).copy()
                    for q in range(k-1):
                        p=k-1-q
                        temp=ShootTemplateFromVectorFieldsFinal(vector_field_list,zeta_list[p],p,k)
                        G[k]+= (functional.inv_N * grad_op(temp)).copy()

                for j in range(functional.nb_data):
                    conv=functional.ConvolveIntegrate(functional.S[j].gradient(image_list[j]),G,j,vector_field_list,zeta_list)
                    for k in range(functional.k_j_list[j]):
                        h[k]-=conv[0][k]
                        eta[k]-=conv[1][k]

                for k in range(functional.N):
                    h[k]+=2*functional.lamb*vector_field_list[k]
                    eta[k]+=2*functional.tau*zeta_list[k]

                return [h,eta]

        return TemporalAttachmentMetamorphosisGeomGradient()












