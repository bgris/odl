#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 13:33:47 2017

@author: bgris
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 09:42:21 2017

@author: bgris
"""
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
from odl.deform import TemporalAttachmentLDDMMGeom
from odl.deform import ShootTemplateFromVectorFields
from odl.solvers.functional.functional import Functional
__all__ = ('TemporalAttachmentLDDMMGeomGatingFunction', 'RegularityLDDMMGatingFunction')



class TemporalAttachmentLDDMMGeomGatingFunction(Functional):

    """Deformation operator with fixed template acting on displacement fields.



        Compute the attachment term of the deformed template, copared to the
        given data at each time point.


    """


    def __init__(self, T, a,nb_time_point_int_gate,nb_time_point_int, template, data, data_time_points, forward_operators,Norm, kernel, domain=None):
        """
        Parameters
        ----------
        T : float between 0 and 1
            the maximal time for the physiological time
        a : real function : [0,1] -> [0,T]
            Gating function for the motion
        nb_time_point_int_gate : int
           number of time points for the numerical integration
           between 0 and T
        nb_time_point_int : int
           number of time points for the numerical integration
           between 0 and 1
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
        self.T=T

        self.a=a

        self.N_tot=nb_time_point_int # number of time points between 0 and 1

        self.N=nb_time_point_int_gate # number of time points between 0 and T

        # temporal time index function : [0,N_tot] -> [0,N] such that
        # a_index(i) *T/N <= a(i/N_tot) < ( a_index(i) +1 ) *T/N
        # Takes values between 0 and N-1
        def a_index(i):
            u=int(self.a(i/self.N_tot) *self.N / self.T)
            return u

        self.a_index=a_index


        TemporalAttachmentLDDMM=TemporalAttachmentLDDMMGeom(nb_time_point_int,
                                                            template, data, data_time_points, forward_operators,Norm, kernel)


        self.TemporalAttachmentLDDMM=TemporalAttachmentLDDMM

        # the domain is lists of vector fields
        if domain is None:
            domain = odl.ProductSpace(
                    template.space.real_space.tangent_bundle,self.N)


        super().__init__(domain,linear=False)


    def _call(self, vector_field_list, out=None):
        # the vector_field_list corresponds to a time-varying
        # vector field defined on [0,T] : N time points

        # we define the corresponding list corresponding to the periodic
        # vector field defined on [0,1]

        vector_field_list_tot = self.TemporalAttachmentLDDMM.domain.element()

        for i in range(self.N_tot +1):
            k_i=self.a_index(i)
            # Compute weighted mean between values at k_i and k_i +1 of
            # vector_field_list

            # ratio of the difference between projected time and
            # the previous integration time point
            delta0 = self.N*(self.a(i/self.N_tot) - k_i*self.T/self.N)

            if (k_i==self.N-1): # in this case : mean between k_i and 0
                vector_field_list_tot[i] =(1-delta0)*vector_field_list[k_i].copy()+ delta0*vector_field_list[0].copy()
            else:
                vector_field_list_tot[i] =(1-delta0)*vector_field_list[k_i].copy() +delta0*vector_field_list[k_i +1].copy()


        return self.TemporalAttachmentLDDMM(vector_field_list_tot)


    @property
    def ComputeTotalVectorFields(self):
        functional = self

        class TotalVectorFields(Operator):

            def __init__(self):
                """Initialize a new instance."""
                super().__init__(functional.domain, functional.TemporalAttachmentLDDMM.domain,
                                 linear=True)

            def _call(self, vector_field_list):
                vector_field_list_tot = functional.TemporalAttachmentLDDMM.domain.element()

                for i in range(functional.N_tot +1):
                    k_i=functional.a_index(i)
                    # Compute weighted mean between values at k_i and k_i +1 of
                    # vector_field_list

                    # ratio of the difference between projected time and
                    # the previous integration time point
                    delta0 = functional.N*(functional.a(i/functional.N_tot) - k_i*functional.T/functional.N)

                    if (k_i==functional.N-1): # in this case : mean between k_i and 0
                        vector_field_list_tot[i] =(1-delta0)*vector_field_list[k_i].copy()+ delta0*vector_field_list[0].copy()
                    else:
                        vector_field_list_tot[i] = (1-delta0)*vector_field_list[k_i].copy() + delta0*vector_field_list[k_i +1].copy()

                return vector_field_list_tot

        return TotalVectorFields()


    @property
    def gradient(self):

        functional = self

        class TemporalAttachmentLDDMMGeomGatingFunctionGradient(Operator):

            """The gradient operator of the TemporalAttachmentLDDMMGeomGatingFunction
            functional."""

            def __init__(self):
                """Initialize a new instance."""
                super().__init__(functional.domain, functional.domain,
                                 linear=False)

            def _call(self, vector_field_list):
                # the vector_field_list corresponds to a time-varying
                # vector field defined on [0,T] : N time points

                # we define the corresponding list corresponding to the periodic
                # vector field defined on [0,1]

                vector_field_list_tot = functional.ComputeTotalVectorFields(
                        vector_field_list)

                grad = functional.TemporalAttachmentLDDMM.gradient(vector_field_list_tot)

                grad_gate=functional.domain.zero()

                for i in range(functional.N_tot +1):
                    k_i=functional.a_index(i)
                    # Compute weighted mean between values at k_i and k_i +1 of
                    # vector_field_list

                    # ratio of the difference between projected time and
                    # the previous integration time point
                    delta0 = functional.N*(functional.a(i/functional.N_tot) - k_i*functional.T/functional.N)

                    if (k_i==functional.N-1): # in this case : mean between k_i and 0
                        grad_gate[k_i]+=(1-delta0)*grad[i].copy()
                        grad_gate[0]+=(delta0)*grad[i].copy()
                    else:
                        grad_gate[k_i]+=(1-delta0)*grad[i].copy()
                        grad_gate[k_i+1]+=(delta0)*grad[i].copy()


                return grad_gate

        return TemporalAttachmentLDDMMGeomGatingFunctionGradient()



class RegularityLDDMMGatingFunction(Functional):



    def __init__(self,T, a,nb_time_point_int_gate, nb_time_point_int, kernel, domain):
        """
        Parameters
        ----------
        T : float between 0 and 1
            the maximal time for the physiological time
        a : real function : [0,1] -> [0,T]
            Gating function for the motion
        nb_time_point_int_gate : int
           number of time points for the numerical integration
           between 0 and T
        nb_time_point_int : int
           number of time points for the numerical integration
           between 0 and 1
        kernel : 'function'
            Kernel function in RKHS.
        domain : Product space
            space of the discretized time-dependent vector fields

        """

        self.T=T
        # temporal time function to project [0,1] to [0,T]

        self.a=a

        self.N_tot=nb_time_point_int # number of time points between 0 and 1
        self.N=nb_time_point_int_gate

        # temporal time index function : [0,N_tot] -> [0,N] such that
        # a_index(i) *T/N <= a(i/N_tot) < ( a_index(i) +1 ) *T/N
        # Takes values between 0 and N-1
        def a_index(i):
            u=int(self.a(i/self.N_tot) *self.N / self.T)
            return u

        self.a_index=a_index


        super().__init__(domain,linear=False)



    @property
    def gradient(self):
        """Gradient operator of the Rosenbrock functional."""
        functional = self

        class RegularityLDDMMGatingFunctionGradient(Operator):

            """The gradient operator of the RegularityLDDMM
            functional."""

            def __init__(self):
                """Initialize a new instance."""
                super().__init__(functional.domain, functional.domain,
                                 linear=False)

            def _call(self, vector_field_list):
                grad = functional.domain.zero()

                for i in range(functional.N_tot +1):
                    k_i=self.a_index(i)
                    # Compute weighted mean between values at k_i and k_i +1 of
                    # vector_field_list

                    # ratio of the difference between projected time and
                    # the previous integration time point
                    delta0 = self.N*(self.a(i/self.N_tot) - k_i*self.T/self.N)

                    if (k_i==self.N-1): # in this case : mean between k_i and 0
                        grad[k_i]+=(1-delta0)*vector_field_list[i].copy()
                        grad[0]+=(delta0)*vector_field_list[i].copy()
                    else:
                        grad[k_i]+=(1-delta0)*vector_field_list[i].copy()
                        grad[k_i+1]+=(delta0)*vector_field_list[i].copy()


                return grad.copy()

        return RegularityLDDMMGatingFunctionGradient()




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
        """Gradient operator of the Rosenbrock functional."""
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











