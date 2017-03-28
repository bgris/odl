#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 10:42:29 2017

@author: bgris
"""


"""Usefull functions for deformations defined thanks to control points."""

# Imports for common Python 2/3 codebase
#from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super

from odl.solvers.functional.functional import Functional
import numpy as np
import odl
from odl.discr import DiscreteLp, Gradient, Divergence
from odl.operator import Operator, PointwiseInner
from odl.space import ProductSpace
from odl.space import rn
from odl.space.fspace import FunctionSpace
__all__ = ('TripleLinearComb','RKHSNorm')

class TripleLinearComb(Operator):
    """ Operator that takes in input weights
    and returns the linear cobination of ElementsToCombine """

    def __init__(self,N1,N2,N3,ElementsToCombine,domain=None):
        """Initialize a new instance.

        Parameters
        ----------
        ElementsToCombine : is a product of products of products
        N1,N2,N3 : are the sizes of the lists
        Other Parameters
        ----------------


        Notes
        -----

        """

        # TODO : vérifier que ElementsToCombine est de la bonne forme

        self.__ElementsToCombine=ElementsToCombine
        self.__N1=N1
        self.__N2=N2
        self.__N3=N3
        domain=odl.ProductSpace(odl.ProductSpace(odl.rn(N3), N2), N1)
        self.__domain=domain
        range_space = ElementsToCombine[0][0][0].space
        self.__range_space=range_space

        super().__init__(domain, range_space, linear=True)


    @property
    def N1(self):
        """Implementation back-end for evaluation of this operator."""
        return self.__N1

    @property
    def N2(self):
        """Implementation back-end for evaluation of this operator."""
        return self.__N2

    @property
    def N3(self):
        """Implementation back-end for evaluation of this operator."""
        return self.__N3


    @property
    def ElementsToCombine(self):
        """Implementation back-end for evaluation of this operator."""
        return self.__ElementsToCombine



    def _call(self,x,out=None):
        Combination=self.__range_space.element(self.__ElementsToCombine[0][0][0])
        Combination=x[0][0][0]*Combination
        for k1 in range (0,self.__N1):
            for k2 in range (0,self.__N2):
                for k3 in range (0,self.__N3):
                    if not (k1==0 and k2==0 and k3==0):
                        Combination=Combination + x[k1][k2][k3]*self.__ElementsToCombine[k1][k2][k3]

        return Combination

    @property
    def adjoint(self):
        operator=self
        class ComputeAdjoint(Operator):
            def __init__(self):
                super().__init__(domain=operator.range,
                 range=operator.domain,
                 linear=True)

            def _call(self,X):
                adj=operator.domain.zero()
                for k1 in range (0,operator.N1):
                    for k2 in range (0,operator.N2):
                        for k3 in range (0,operator.N3):
                            adj[k1][k2][k3]=X.inner(
                            operator.ElementsToCombine[k1][k2][k3])

                return adj

        return ComputeAdjoint()


class RKHSNorm(Functional):
    """ Computes the norm of the generated vector field"""

    def __init__(self,scale,NbCP,space,  domain=None):
        self.__scale=scale
        self.__NbCP=NbCP
        super().__init__(space=space, linear=False)

    @property
    def scale(self):
        """Fixed scale of this deformation operator."""
        return self.__scale

    @property
    def NbCP(self):
        """Fixed scale of this deformation operator."""
        return self.__NbCP

    def _call(self,x,out=None):
        CP=x[0]
        MOM=x[1]
        Energy_reg=0
        for k in range(0,self.__NbCP):
            for j in range(0,self.__NbCP):
                prod= MOM[k][0]* MOM[j][0] +MOM[k][1]* MOM[j][1]
                Energy_reg=Energy_reg+0.5*prod*np.exp(-((CP[j][0]-CP[k][0]) ** 2 + (CP[j][1]-CP[k][1])  ** 2) / (2 * self.__scale ** 2))
        return Energy_reg

    @property
    def gradient(self):

        functional = self
        class Compute(Operator):

            """The gradient operator of this functional."""

            def __init__(self):
                """Initialize a new instance."""
                super().__init__(functional.domain, functional.domain,
                         linear=False)

            def _call(self,X):
                """Apply the gradient operator to the given point."""
                ndim=2
                NbCP=functional.NbCP
                sigma=functional.scale
                CP=X[0]
                MOM=X[1]
                GradCP=odl.ProductSpace(odl.rn(ndim), NbCP).element()
                GradMOM=odl.ProductSpace(odl.rn(ndim), NbCP).element()
                for k in range(0,NbCP):
                    for j in range(0,NbCP):
                        if(k!=j):
                            GradMOM[k][0]=GradMOM[k][0]+MOM[j][0]*np.exp(-((CP[j][0]-CP[k][0]) ** 2 + (CP[j][1]-CP[k][1])  ** 2) / (2 * sigma ** 2))
                            GradMOM[k][1]=GradMOM[k][1]+MOM[j][1]*np.exp(-((CP[j][0]-CP[k][0]) ** 2 + (CP[j][1]-CP[k][1])  ** 2) / (2 * sigma ** 2))
                            prod= MOM[k][0]* MOM[j][0] +MOM[k][1]* MOM[j][1]
                            GradCP[k][0]=GradCP[k][0]+((CP[k][0]-CP[j][0])/(sigma ** 2))*prod* np.exp(-((CP[j][0]-CP[k][0]) ** 2 + (CP[j][1]-CP[k][1])  ** 2) / (2 * sigma ** 2))
                            GradCP[k][1]=GradCP[k][1]+((CP[k][1]-CP[j][1])/(sigma ** 2))*prod* np.exp(-((CP[j][0]-CP[k][0]) ** 2 + (CP[j][1]-CP[k][1])  ** 2) / (2 * sigma ** 2))
                        else:
                            GradMOM[k][0]=GradMOM[k][0]+MOM[j][0]*np.exp(-((CP[j][0]-CP[k][0]) ** 2 + (CP[j][1]-CP[k][1])  ** 2) / (2 * sigma ** 2))
                            GradMOM[k][1]=GradMOM[k][1]+MOM[j][1]*np.exp(-((CP[j][0]-CP[k][0]) ** 2 + (CP[j][1]-CP[k][1])  ** 2) / (2 * sigma ** 2))

                return odl.ProductSpace(odl.ProductSpace(odl.rn(ndim), NbCP),
                                2).element([GradCP,GradMOM])


        return Compute()
