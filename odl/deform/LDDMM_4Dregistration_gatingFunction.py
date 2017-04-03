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


    def __init__(self, a,nb_time_point_int, template, data, data_time_points, forward_operators,Norm, kernel, domain=None):
        """
        Parameters
        ----------
        a : real function : [0,1] -> [0,T]
            Gating functionfor the motion
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

        # we need to define the corresponding number of time points for
        # the numerical integration between 0 and T : it is the number of
        # time interval 1/N that can be put in [0,T]
        k=int(self.N_tot * self.T)
        self.N=k # number of time points between 0 and T

        # temporal time index function to project indexes in [0,N_tot+1] to
        # indexes in [0,N]
        def a_index(i):
            k= int(i/self.N)
            u=int(i-k*self.N) -1
            return int(u)

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
            vector_field_list_tot[i]=vector_field_list[self.a_index(i)].copy()

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
                    vector_field_list_tot[i]=vector_field_list[functional.a_index(i)].copy()

                return vector_field_list_tot

        return TotalVectorFields()


    @property
    def gradient(self):

        functional = self

        class TemporalAttachmentLDDMMGeomPeriodicGradient(Operator):

            """The gradient operator of the TemporalAttachmentLDDMMGeom
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

                vector_field_list_tot = functional.TemporalAttachmentLDDMM.domain.element()

                for i in range(functional.N_tot +1):
                    vector_field_list_tot[i]=vector_field_list[functional.a_index(i)].copy()

                grad = functional.TemporalAttachmentLDDMM.gradient(vector_field_list_tot)

                grad_per=functional.domain.zero()

                for i in range(functional.N_tot +1):
                    grad_per[functional.a_index(i)]+=grad[i].copy()

                return grad_per

        return TemporalAttachmentLDDMMGeomPeriodicGradient()



class RegularityLDDMMPeriodic(Functional):



    def __init__(self, T, nb_time_point_int, kernel, domain):
        """
        Parameters
        ----------
        T : float between 0 and 1
            the period of the motion
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
        def a(t):
            k= int(t/self.T)
            return t-k*T

        self.a=a

        self.N_tot=nb_time_point_int # number of time points between 0 and 1

        # we need to define the corresponding number of time points for
        # the numerical integration between 0 and T : it is the number of
        # time interval 1/N that can be put in [0,T]
        k=int(self.N_tot * self.T) # number of period between 0 and 1
        self.N=k # number of time points between 0 and T

        # temporal time index function to project indexes in [0,N_tot+1] to
        # indexes in [0,N]
        def a_index(i):
            k= int(i/self.N)
            u=int(i-k*self.N) -1
            return int(u)

        self.a_index=a_index


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
                grad = functional.domain.zero()

                for i in range(functional.N_tot +1):
                    grad[functional.a_index(i)]+=vector_field_list[functional.a_index(i)].copy()

                return grad.copy()

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











