#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 10:44:42 2017

@author: bgris
"""


"""Class to generate small deformations defined thanks to control points."""

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
__all__ = ('CPVectorField',)


class CPVectorField(Operator):
    """ Operator that associates to each (CP,MOM) a vector fields on the space
    disp_field_space """



    def __init__(self,scale,NbCP, space, domain=None):
        """Initialize a new instance.

        Parameters
        ----------
        sigma : Non negative real number
            scale of the deformation
        NbCP : Integer
            Number of control points
        template : `DiscreteLpElement`
            Fixed template that is to be deformed.
        domain : power space of `DiscreteLp`, optional
            The space of all allowed coordinates in the deformation.
            A `ProductSpace` of ``template.ndim`` copies of a function-space.
            It must fulfill
            ``domain[0].partition == template.space.partition``, so
            this option is useful mainly when using different interpolations
            in the displacement and template.
            Default: ``template.space.real_space.tangent_bundle``

        Examples
        --------

        """

        if (scale <= 0):
            raise TypeError('sigma must be a positive real number')
        self.__scale=scale

        if not ((isinstance(NbCP,int) and NbCP >= 0)):
            raise TypeError('NbCP must be a positive integer')
        self.__NbCP=NbCP

        self.__space=space
        self.__disp_field_space=space.tangent_bundle

        if domain is None:
            domain = odl.ProductSpace(odl.ProductSpace(odl.rn(self.__space.ndim), NbCP), odl.ProductSpace(odl.rn(self.__space.ndim), NbCP))
#        else:
#            if not isinstance(domain, ProductSpace):
#                raise TypeError('`domain` must be a `ProductSpace` '
#                                'instance, got {!r}'.format(domain))
#            if not domain.is_power_space:
#                raise TypeError('`domain` must be a power space, '
#                                'got {!r}'.format(domain))
#            if not isinstance(domain[0], DiscreteLp):
#                raise TypeError('`domain[0]` must be a `DiscreteLp` '
#                                'instance, got {!r}'.format(domain[0]))
#
#            if template.space.partition != domain[0].partition:
#                raise ValueError(
#                    '`template.space.partition` not equal to `coord_space`s '
#                    'partiton ({!r} != {!r})'
#                    ''.format(template.space.partition, domain[0].partition))

        self.__domain=domain
        super().__init__(domain=domain,
                         range=self.__disp_field_space,
                         linear=False)


    @property
    def space(self):
        """Fixed template of this deformation operator."""
        return self.__space

    @property
    def domain(self):
        """Fixed template of this deformation operator."""
        return self.__domain

    @property
    def disp_field_space(self):
        """Fixed template of this deformation operator."""
        return self.__disp_field_space

    @property
    def NbCP(self):
        """Number of control points of this deformation operator."""
        return self.__NbCP

    @property
    def scale(self):
        """Number of control points of this deformation operator."""
        return self.__scale


    def _call(self,x,out=None):
        CP=x[0]
        MOM=x[1]
        if not (len(CP)==self.__NbCP):
            raise TypeError('Size of input CP is {} instead of {}'.format(len(CP),self.__NbCP ))
        if not (len(MOM)==self.__NbCP):
            raise TypeError('Size of input MOM is {} instead of {}'.format(len(MOM),self.__NbCP ))


        def VX(x):
            a=0
            for k in range(0,self.__NbCP):
                a=a+MOM[k][0]* np.exp(-((x[0]-CP[k][0]) ** 2 + (x[1]-CP[k][1])  ** 2) / (2 * self.__scale ** 2))
            return a

        def VY(x):
            a=0
            for k in range(0,self.__NbCP):
                a=a+MOM[k][1]* np.exp(-((x[0]-CP[k][0]) ** 2 + (x[1]-CP[k][1])  ** 2) / (2 * self.__scale ** 2))
            return a

        disp_func = [VX,VY]

        return self.__disp_field_space.element(disp_func)

#    @property
    def derivative(self,X):
#        derivative_op=CPVectorFieldDer(self.__scale,self.__NbCP, self.__space, domain=None)
#        return derivative_op(X)
#        functional = self
#        class  ComputeDer(Operator):
#            def __init__(self):
#                super().__init__(domain=functional.domain, range=functional.range,  linear=False)
#
#            def _call(self,X):
        CP=X[0]
        MOM=X[1]
        if not (len(CP)==self.NbCP):
            raise TypeError('Size of input CP is {} instead of {}'.format(len(CP),self.NbCP ))
        if not (len(MOM)==self.NbCP):
            raise TypeError('Size of input MOM is {} instead of {}'.format(len(MOM),self.NbCP ))

        disp_field_space = self.disp_field_space
        derivateCP=odl.ProductSpace(odl.ProductSpace(disp_field_space, self.space.ndim),  self.NbCP).element()
        derivateMOM=odl.ProductSpace(odl.ProductSpace(disp_field_space, self.space.ndim),  self.NbCP).element()
        for k in range(0,self.NbCP):
            disp_func_grad = [
                              lambda x: ((x[0]-CP[k][0])/(self.scale ** 2))*MOM[k][0]* np.exp(-((x[0]-CP[k][0]) ** 2 + (x[1]-CP[k][1])  ** 2) / (2 * self.scale ** 2)),
                              lambda x: ((x[0]-CP[k][0])/(self.scale ** 2))*MOM[k][1]* np.exp(-((x[0]-CP[k][0]) ** 2 + (x[1]-CP[k][1])  ** 2) / (2 * self.scale ** 2))]

            disp_field_grad = disp_field_space.element(disp_func_grad)
            derivateCP[k][0]=disp_field_grad

            # Derivative of disp_field_est with respect to CP[k][1]
            disp_func_grad = [
                              lambda x: ((x[1]-CP[k][1])/(self.scale ** 2))*MOM[k][0]* np.exp(-((x[0]-CP[k][0]) ** 2 + (x[1]-CP[k][1])  ** 2) / (2 * self.scale ** 2)),
                              lambda x: ((x[1]-CP[k][1])/(self.scale ** 2))*MOM[k][1]* np.exp(-((x[0]-CP[k][0]) ** 2 + (x[1]-CP[k][1])  ** 2) / (2 * self.scale ** 2))]
            disp_field_grad = disp_field_space.element(disp_func_grad)
            derivateCP[k][1]=disp_field_grad

            # Derivative of disp_field_est with respect to MOM[k][0]
            disp_func_grad = [
                                  lambda x: np.exp(-((x[0]-CP[k][0]) ** 2 + (x[1]-CP[k][1])  ** 2) / (2 * self.scale ** 2)),
                                  lambda x: 0]
            disp_field_grad = disp_field_space.element(disp_func_grad)
            derivateMOM[k][0]=disp_field_grad

            # Derivative of disp_field_est with respect to MOM[k][1]
            disp_func_grad = [
            lambda x: 0,
            lambda x: np.exp(-((x[0]-CP[k][0]) ** 2 + (x[1]-CP[k][1])  ** 2) / (2 * self.scale ** 2))]
            disp_field_grad = disp_field_space.element(disp_func_grad)
            derivateMOM[k][1]=disp_field_grad

        derivate= odl.ProductSpace(odl.ProductSpace(odl.ProductSpace(disp_field_space, self.space.ndim),  self.NbCP),2).element([derivateCP,derivateMOM])

        return odl.deform.TripleLinearComb(2,self.NbCP,self.space.ndim,derivate)
