#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 11:40:30 2017

@author: bgris
"""

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

"""Abstract class for small deformations defined thanks to control points."""

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
__all__ = ('TripleLinearComb','CPVectorField','CPVectorFieldActionFixedTemplate','RKHSNorm','CPSmallDeformFixedTempl')

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

        return TripleLinearComb(2,self.space.ndim,self.NbCP,derivate)

#            @property


#        return ComputeDer()

class CPVectorFieldActionFixedTemplate(Operator):
    """ Operator that has a fixed template and associates to each (CP,MOM) a
     the image of the template deformed by the small deformation parametrized
     by (CP,MOM) """



    def __init__(self,scale,NbCP, template, domain=None):
        """Initialize a new instance.

        Parameters
        ----------
        sigma : Non negative real number
            scale of the deformation
        NbCP : Integer
            Number of control points

        Examples
        --------

        """

        if (scale <= 0):
            raise TypeError('sigma must be a positive real number')
        self.scale=scale

        if not ((isinstance(NbCP,int) and NbCP >= 0)):
            raise TypeError('NbCP must be a positive integer')

        self.NbCP=NbCP
        self.template=template
        self.space=template.space
        self.disp_field_space=template.space.tangent_bundle
        self.vectorfield_op=CPVectorField(scale,NbCP,template.space,domain=domain)
        self.deform_op = odl.deform.LinDeformFixedTemplForward(template)

        if domain is None:
            domain = odl.ProductSpace(odl.ProductSpace(odl.rn(self.space.ndim),
                                                       NbCP), 2)
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
                         range=self.deform_op.range,
                         linear=False)


    def _call(self,X,out=None):
        deform_op_CP=self.deform_op*self.vectorfield_op
        return deform_op_CP(X)

    @property
    def derivative(self):
        deform_op_CP=self.deform_op*self.vectorfield_op
        return deform_op_CP.derivative



#
#        functional=self
#        class Compute(Operator):
#            def __init__(self):
#                """Initialize a new instance."""
#                self.NbCP=functional.NbCP
#                self.sigma=functional.scale
#                super().__init__(functional.domain, functional.domain,
#                         linear=False)
#
#            def _call(self,X):
#                deform_op_CP=functional.deform_op*functional.vectorfield_op
#                return deform_op_CP.derivative(X)
#
#        return Compute()
#
#            def adjoint(self,X):
#                template_der=functional.template.space.tangent_bundle.element()
#                functionalAdjoint=self
#                class ComputeAdjoint(self,Y):
#                    def __init__(self):
#                        """Initialize a new instance."""
#                        super().__init__(functionalAdjoint.domain,
#                                 functionalAdjoint.domain,
#                                 linear=False)
#
#                    def _call(self,Image):
#                        NbCP=functionalAdjoint.NbCP
#                        sigma=functionalAdjoint.scale
#                        GradCP=odl.ProductSpace(odl.rn(ndim), NbCP).element()
#                        GradMOM=odl.ProductSpace(odl.rn(ndim), NbCP).element()
#



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



class CPSmallDeformFixedTempl(Functional):

    """Deformation operator with fixed template acting on Control Points and Momenta (which tmeselves define a vector field).
    This operator also has a fixed data attachment function (in general defined thanks to a fixed target)
    The operator has a fixed template ``I`` and maps a displacement
    field ``v`` to the new function ``x --> I(x + v(x))``.

    See Also
    --------



    Notes
    -----

    """

    def __init__(self,vector_field_op,deform_op,obj_fun,metric, domain=None):
        """Initialize a new instance.

        Parameters
        ----------
        deform_op : deformation operator
            says how the template is deformed frome a vector field
        vector_field_op : vector field operator
            says how vector fields are generated

        Examples
        --------

        """

        self.__deform_op=deform_op
        self.__vector_field_op=vector_field_op
        self.__obj_fun=obj_fun
        self.__composed_op=self.__obj_fun*self.__deform_op*self.__vector_field_op
        self.__space=deform_op.template.space
        self.metric=metric

        if domain is None:
#            domain = odl.ProductSpace(odl.ProductSpace(odl.rn(self.__space.ndim), vector_field_op.NbCP), odl.ProductSpace(odl.rn(self.__space.ndim), vector_field_op.NbCP))
            domain=vector_field_op.domain
        super().__init__(space=domain,
                         linear=False)

    @property
    def deform_op(self):
        """Fixed scale of this deformation operator."""
        return self.__deform_op

    def space(self):
        """Fixed scale of this deformation operator."""
        return self.__space

    def obj_fun(self):
        """Fixed scale of this deformation operator."""
        return self.__obj_fun

    def vector_field_op(self):
        """Fixed template of this deformation operator."""
        return self.__vector_field_op

    def metric(self):
        """Fixed template of this deformation operator."""
        return self.__metric

    def _call(self, X, out=None):
        """Implementation of ``self(displacement[, out])``."""
        return self.__composed_op(X, out)+ self.__metric(X)

    def ComputeDeformTemplate(self,X):
        deform_op_CP=self.__deform_op*self.__vector_field_op
        return deform_op_CP(X)


    def derivative(self, X):
        """Derivative of the operator at ``X=[CP,MOM]``.

        Parameters
        ----------
        X : `domain` `element-like`
            Point at which the derivative is computed.

        Returns
        -------
        derivative : `PointwiseInner`
            The derivative evaluated at ``X``.
        """
        # To implement the complex case we need to be able to embed the real
        # vector field space into the range of the gradient. Issue #59.
        if not self.range.is_rn:
            raise NotImplementedError('derivative not implemented for complex '
                                      'spaces.')

        return self.__composed_op.derivative(X)

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
                NbCP=functional.vector_field_op().NbCP
                sigma=functional.vector_field_op().scale
                GradCP=odl.ProductSpace(odl.rn(ndim), NbCP).element()
                GradMOM=odl.ProductSpace(odl.rn(ndim), NbCP).element()
                CP=X[0]
                MOM=X[1]
        #        der=functional.derivative(X)
                disp_field_est=functional.vector_field_op()(X)
                deformed_template_est=functional.deform_op(disp_field_est)
                GradCP=odl.ProductSpace(odl.rn(ndim), NbCP).element()
                GradMOM=odl.ProductSpace(odl.rn(ndim), NbCP).element()
                for k in range(0,NbCP):
                    disp_func_grad = [
                                      lambda x: ((x[0]-CP[k][0])/(sigma ** 2))*MOM[k][0]* np.exp(-((x[0]-CP[k][0]) ** 2 + (x[1]-CP[k][1])  ** 2) / (2 * sigma ** 2)),
                                      lambda x: ((x[0]-CP[k][0])/(sigma ** 2))*MOM[k][1]* np.exp(-((x[0]-CP[k][0]) ** 2 + (x[1]-CP[k][1])  ** 2) / (2 * sigma ** 2))]

                    disp_field_grad = functional.vector_field_op().disp_field_space.element(disp_func_grad)
                    # Derivative of the application of the vector field to template with respect to CP[k][0]
                    deform_op_deriv = functional.deform_op.derivative(disp_field_est)(disp_field_grad)
                    # Derivation of the energy with respect to CP[0]
                    GradCP[k][0]=functional.obj_fun().derivative(deformed_template_est)(deform_op_deriv)


                    # Derivative of disp_field_est with respect to CP[k][1]
                    disp_func_grad = [
                                      lambda x: ((x[1]-CP[k][1])/(sigma ** 2))*MOM[k][0]* np.exp(-((x[0]-CP[k][0]) ** 2 + (x[1]-CP[k][1])  ** 2) / (2 * sigma ** 2)),
                                      lambda x: ((x[1]-CP[k][1])/(sigma ** 2))*MOM[k][1]* np.exp(-((x[0]-CP[k][0]) ** 2 + (x[1]-CP[k][1])  ** 2) / (2 * sigma ** 2))]
                    disp_field_grad = functional.vector_field_op().disp_field_space.element(disp_func_grad)
                    # Derivative of the application of the vector field to template with respect to CP[k][0]
                    deform_op_deriv = functional.deform_op.derivative(disp_field_est)(disp_field_grad)
                    # Derivation of the energy with respect to CP[0]
                    GradCP[k][1]=functional.obj_fun().derivative(deformed_template_est)(deform_op_deriv)


                    # Derivative of disp_field_est with respect to MOM[k][0]
                    disp_func_grad = [
                                          lambda x: np.exp(-((x[0]-CP[k][0]) ** 2 + (x[1]-CP[k][1])  ** 2) / (2 * sigma ** 2)),
                                          lambda x: 0]
                    disp_field_grad = functional.vector_field_op().disp_field_space.element(disp_func_grad)
                    # Derivative of the application of the vector field to template with respect to CP[k][0]
                    deform_op_deriv = functional.deform_op.derivative(disp_field_est)(disp_field_grad)
                    # Derivation of the energy with respect to CP[0]
                    GradMOM[k][0]=functional.obj_fun().derivative(deformed_template_est)(deform_op_deriv)


                    # Derivative of disp_field_est with respect to MOM[k][1]
                    disp_func_grad = [
                    lambda x: 0,
                    lambda x: np.exp(-((x[0]-CP[k][0]) ** 2 + (x[1]-CP[k][1])  ** 2) / (2 * sigma ** 2))]
                    disp_field_grad = functional.vector_field_op().disp_field_space.element(disp_func_grad)
                    # Derivative of the application of the vector field to template with respect to CP[k][0]
                    deform_op_deriv = functional.deform_op.derivative(disp_field_est)(disp_field_grad)
                    # Derivation of the energy with respect to CP[0]
                    GradMOM[k][1]=functional.obj_fun().derivative(deformed_template_est)(deform_op_deriv)

                A=functional.metric().gradient()(X)
                B=odl.ProductSpace(odl.ProductSpace(odl.rn(ndim), NbCP),
                                2).element([GradCP,GradMOM])
                return A+B

        return Compute()






    def __repr__(self):
        """Return ``repr(self)``."""
        arg_reprs = [repr(self.template)]
#        if self.domain != self.__displacement.space[0]:
#            arg_reprs.append('domain={!r}'.format(self.domain))
        arg_str = ', '.join(arg_reprs)

        return '{}({})'.format(self.__class__.__name__, arg_str)




#
#def _generate_disp_field(template, scale, NbCP, CP, MOM):
#    if not (len(CP)==NbCP):
#        raise TypeError('Size of input CP is {} instead of {}'.format(len(CP),NbCP ))
#    if not (len(MOM)==NbCP):
#        raise TypeError('Size of input MOM is {} instead of {}'.format(len(MOM),NbCP ))
#
#    disp_field_space = template.space.tangent_bundle
#    def VX(x):
#        a=0
#        for k in range(0,NbCP):
#            a=a+MOM[k][0]* np.exp(-((x[0]-CP[k][0]) ** 2 + (x[1]-CP[k][1])  ** 2) / (2 * scale ** 2))
#        return a
#
#    def VY(x):
#        a=0
#        for k in range(0,NbCP):
#            a=a+MOM[k][1]* np.exp(-((x[0]-CP[k][0]) ** 2 + (x[1]-CP[k][1])  ** 2) / (2 * scale ** 2))
#        return a
#
#    disp_func = [VX,VY]
#
#    return disp_field_space.element(disp_func)
#
#
#def _generate_disp_field_derivate(template, scale, NbCP,CP,MOM):
#    # Returns an operator equal to the derivate of the generated vector field at [CP,MOM]
#    # The intermediate computation tool derivate is list of two lists of vector fields
#    # derivate[i][j][u] is the derivative w.r.t the u-th component of the j-th CP/MOM if i equals 0/1 (it is a list of 2 matrices, each one is the X/Y component of the displacement)
#    if not (len(CP==self.__NbCP)):
#        raise TypeError('Size of input CP is {} instead of {}'.format(len(CP),self.__NbCP ))
#    if not (len(MOM==self.__NbCP)):
#        raise TypeError('Size of input MOM is {} instead of {}'.format(len(MOM),self.__NbCP ))
#
#
#    disp_field_space = self.__template.space.tangent_bundle
#    derivateCP=odl.ProductSpace(odl.ProductSpace(disp_field_space, self.__template.space.ndim),  self.__NbCP).element()
#    derivateMOM=odl.ProductSpace(odl.ProductSpace(disp_field_space, self.__template.space.ndim),  self.__NbCP).element()
#    for k in range(0,NbCP):
#        disp_func_grad = [
#                          lambda x: ((x[0]-CP[k][0])/(sigma ** 2))*MOM[k][0]* np.exp(-((x[0]-CP[k][0]) ** 2 + (x[1]-CP[k][1])  ** 2) / (2 * sigma ** 2)),
#                          lambda x: ((x[0]-CP[k][0])/(sigma ** 2))*MOM[k][1]* np.exp(-((x[0]-CP[k][0]) ** 2 + (x[1]-CP[k][1])  ** 2) / (2 * sigma ** 2))]
#
#        disp_field_grad = disp_field_space.element(disp_func_grad)
#        derivateCP[k][0]=disp_field_grad
#
#        # Derivative of disp_field_est with respect to CP[k][1]
#        disp_func_grad = [
#                          lambda x: ((x[1]-CP[k][1])/(sigma ** 2))*MOM[k][0]* np.exp(-((x[0]-CP[k][0]) ** 2 + (x[1]-CP[k][1])  ** 2) / (2 * sigma ** 2)),
#                          lambda x: ((x[1]-CP[k][1])/(sigma ** 2))*MOM[k][1]* np.exp(-((x[0]-CP[k][0]) ** 2 + (x[1]-CP[k][1])  ** 2) / (2 * sigma ** 2))]
#        disp_field_grad = disp_field_space.element(disp_func_grad)
#        derivateCP[k][1]=disp_field_grad
#
#        # Derivative of disp_field_est with respect to MOM[k][0]
#        disp_func_grad = [
#                              lambda x: np.exp(-((x[0]-CP[k][0]) ** 2 + (x[1]-CP[k][1])  ** 2) / (2 * sigma ** 2)),
#                              lambda x: 0]
#        disp_field_grad = disp_field_space.element(disp_func_grad)
#        derivateMOM[k][0]=disp_field_grad
#
#        # Derivative of disp_field_est with respect to MOM[k][1]
#        disp_func_grad = [
#        lambda x: 0,
#        lambda x: np.exp(-((x[0]-CP[k][0]) ** 2 + (x[1]-CP[k][1])  ** 2) / (2 * sigma ** 2))]
#        disp_field_grad = disp_field_space.element(disp_func_grad)
#        derivateMOM[k][1]=disp_field_grad
#
#    derivate=self.__domain.element(derivateCP,derivateMOM)
#    # il faudrait retourner un operateur qui prend un Delta(CP,MOM)
#    # et renvoie le champs de vecteurs egal à la combinaison lineaire
#    # des champs de vect de derivate avec comme poids les composantes
#    # de Delta(CP,MOM)
#    return disp_field_space.element(PointwiseInner(self.domain, derivate))
#
#
#
#def _linear_deformForward(template, displacement, out=None):
#    """Linearized deformation of a template with a displacement field.
#
#    The function maps a given template ``I`` and a given displacement
#    field ``v`` to the new function ``x --> I(x + v(x))``.
#
#    Parameters
#    ----------
#    template : `DiscreteLpElement`
#        Template to be deformed by a displacement field.
#    displacement : element of power space of ``template.space``
#        Vector field (displacement field) used to deform the
#        template.
#    out : `numpy.ndarray`, optional
#        Array to which the function values of the deformed template
#        are written. It must have the same shape as ``template`` and
#        a data type compatible with ``template.dtype``.
#
#    Returns
#    -------
#    deformed_template : `numpy.ndarray`
#        Function values of the deformed template. If ``out`` was given,
#        the returned object is a reference to it.
#
#    Examples
#    --------
#    Create a simple 1D template to initialize the operator and
#    apply it to a displacement field. Where the displacement is zero,
#    the output value is the same as the input value.
#    In the 4-th point, the value is taken from 0.2 (one cell) to the
#    left, i.e. 1.0.
#
#    >>> space = odl.uniform_discr(0, 1, 5)
#    >>> disp_field_space = space.tangent_bundle
#    >>> template = space.element([0, 0, 1, 0, 0])
#    >>> displacement_field = disp_field_space.element([[0, 0, 0, -0.2, 0]])
#    >>> _linear_deform(template, displacement_field)
#    array([ 0.,  0.,  1.,  1.,  0.])
#
#    The result depends on the chosen interpolation. With 'linear'
#    interpolation and an offset equal to half the distance between two
#    points, 0.1, one gets the mean of the values.
#
#    >>> space = odl.uniform_discr(0, 1, 5, interp='linear')
#    >>> disp_field_space = space.tangent_bundle
#    >>> template = space.element([0, 0, 1, 0, 0])
#    >>> displacement_field = disp_field_space.element([[0, 0, 0, -0.1, 0]])
#    >>> _linear_deform(template, displacement_field)
#    array([ 0. ,  0. ,  1. ,  0.5,  0. ])
#    """
#    image_pts = template.space.points()
#    for i, vi in enumerate(displacement):
#        image_pts[:, i] -= vi.ntuple.asarray()
#    return template.interpolation(image_pts.T, out=out, bounds_check=False)
#


#
#class CPVectorFieldDer(Operator):
#    """ Operator that associates to each (CP,MOM) a vector fields on the space
#    disp_field_space """
#
#    def __init__(self,scale,NbCP, space, domain=None):
#        """Initialize a new instance.
#
#        Parameters
#        ----------
#        sigma : Non negative real number
#            scale of the deformation
#        NbCP : Integer
#            Number of control points
#        template : `DiscreteLpElement`
#            Fixed template that is to be deformed.
#        domain : power space of `DiscreteLp`, optional
#            The space of all allowed coordinates in the deformation.
#            A `ProductSpace` of ``template.ndim`` copies of a function-space.
#            It must fulfill
#            ``domain[0].partition == template.space.partition``, so
#            this option is useful mainly when using different interpolations
#            in the displacement and template.
#            Default: ``template.space.real_space.tangent_bundle``
#
#        Examples
#        --------
#
#        """
#
#        if (scale <= 0):
#            raise TypeError('sigma must be a positive real number')
#        self.__scale=scale
#
#        if not ((isinstance(NbCP,int) and NbCP >= 0)):
#            raise TypeError('NbCP must be a positive integer')
#        self.__NbCP=NbCP
#
#        self.__space=space
#        self.__disp_field_space=space.tangent_bundle
#
#        if domain is None:
#            self.__domain = odl.ProductSpace(odl.ProductSpace(odl.rn(self.__space.ndim), NbCP), odl.ProductSpace(odl.rn(self.__space.ndim), NbCP))
##        else:
##            if not isinstance(domain, ProductSpace):
##                raise TypeError('`domain` must be a `ProductSpace` '
##                                'instance, got {!r}'.format(domain))
##            if not domain.is_power_space:
##                raise TypeError('`domain` must be a power space, '
##                                'got {!r}'.format(domain))
##            if not isinstance(domain[0], DiscreteLp):
##                raise TypeError('`domain[0]` must be a `DiscreteLp` '
##                                'instance, got {!r}'.format(domain[0]))
##
##            if template.space.partition != domain[0].partition:
##                raise ValueError(
##                    '`template.space.partition` not equal to `coord_space`s '
##                    'partiton ({!r} != {!r})'
##                    ''.format(template.space.partition, domain[0].partition))
#
#        super().__init__(domain=self.__domain,
#                         range=odl.space.fspace.FunctionSpace(self.__domain, field=None, out_dtype=None),
#                         linear=False)
#
#
#    def _call(self,X,out=None):
#        CP=X[0]
#        MOM=X[1]
#        if not (len(CP)==self.__NbCP):
#            raise TypeError('Size of input CP is {} instead of {}'.format(len(CP),self.__NbCP ))
#        if not (len(MOM)==self.__NbCP):
#            raise TypeError('Size of input MOM is {} instead of {}'.format(len(MOM),self.__NbCP ))
#
#        disp_field_space = self.__disp_field_space
#        derivateCP=odl.ProductSpace(odl.ProductSpace(disp_field_space, self.__space.ndim),  self.__NbCP).element()
#        derivateMOM=odl.ProductSpace(odl.ProductSpace(disp_field_space, self.__space.ndim),  self.__NbCP).element()
#        for k in range(0,self.__NbCP):
#            disp_func_grad = [
#                              lambda x: ((x[0]-CP[k][0])/(self.__scale ** 2))*MOM[k][0]* np.exp(-((x[0]-CP[k][0]) ** 2 + (x[1]-CP[k][1])  ** 2) / (2 * self.__scale ** 2)),
#                              lambda x: ((x[0]-CP[k][0])/(self.__scale ** 2))*MOM[k][1]* np.exp(-((x[0]-CP[k][0]) ** 2 + (x[1]-CP[k][1])  ** 2) / (2 * self.__scale ** 2))]
#
#            disp_field_grad = disp_field_space.element(disp_func_grad)
#            derivateCP[k][0]=disp_field_grad
#
#            # Derivative of disp_field_est with respect to CP[k][1]
#            disp_func_grad = [
#                              lambda x: ((x[1]-CP[k][1])/(self.__scale ** 2))*MOM[k][0]* np.exp(-((x[0]-CP[k][0]) ** 2 + (x[1]-CP[k][1])  ** 2) / (2 * self.__scale ** 2)),
#                              lambda x: ((x[1]-CP[k][1])/(self.__scale ** 2))*MOM[k][1]* np.exp(-((x[0]-CP[k][0]) ** 2 + (x[1]-CP[k][1])  ** 2) / (2 * self.__scale ** 2))]
#            disp_field_grad = disp_field_space.element(disp_func_grad)
#            derivateCP[k][1]=disp_field_grad
#
#            # Derivative of disp_field_est with respect to MOM[k][0]
#            disp_func_grad = [
#                                  lambda x: np.exp(-((x[0]-CP[k][0]) ** 2 + (x[1]-CP[k][1])  ** 2) / (2 * self.__scale ** 2)),
#                                  lambda x: 0]
#            disp_field_grad = disp_field_space.element(disp_func_grad)
#            derivateMOM[k][0]=disp_field_grad
#
#            # Derivative of disp_field_est with respect to MOM[k][1]
#            disp_func_grad = [
#            lambda x: 0,
#            lambda x: np.exp(-((x[0]-CP[k][0]) ** 2 + (x[1]-CP[k][1])  ** 2) / (2 * self.__scale ** 2))]
#            disp_field_grad = disp_field_space.element(disp_func_grad)
#            derivateMOM[k][1]=disp_field_grad
#
#        derivate=odl.ProductSpace(odl.ProductSpace(odl.ProductSpace(disp_field_space, self.__space.ndim),  self.__NbCP),2).element([derivateCP,derivateMOM])
#
#        return TripleLinearComb(2,self.__space.ndim,self.__NbCP,derivate)




#if __name__ == '__main__':
#    # pylint: disable=wrong-import-position
#    from odl.util.testutils import run_doctests
#    run_doctests()
