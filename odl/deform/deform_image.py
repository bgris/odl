#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 09:58:52 2017

@author: bgris
"""


"""Classes for applying deformations to a template."""
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
__all__ = ('DeformTemplateGeometric','obj_fun_dir')



class DeformTemplateGeometric(Operator):
    """ deformation of images through geometric action"""
    def __init__(self,template):
#       self.deform_op=deform_op
        self.template=template
        self.nb_pts_grid=template.space.shape[0]*template.space.shape[1]


        super().__init__(domain=odl.ProductSpace(odl.rn(self.nb_pts_grid),2),range=self.template.space,
                         linear=False)

    def _call(self,grid):
#        grid_points_defo=self.deform_op.inverse()
        template_def=self.template.interpolation(odl.ProductSpace(odl.rn(self.nb_pts_grid),2).element(grid),out=None,bounds_check=False)
        return self.template.space.element(template_def)

    def get_used_grid(self,deform_op):
        return deform_op.inverse()

    def deform_from_operator(self,deform_op):
        grid=self.get_used_grid(deform_op)
        return self(grid)

    def apply_vect_field(self,vect_field,grid):
        # corresponds to the derivative of the group action: linear with
        # respect to the vector field
        grid_velocity=self.domain.element()
        A=vect_field[0].interpolation(odl.ProductSpace(odl.rn(self.nb_pts_grid),2).element(grid),out=None,bounds_check=False)
        B=vect_field[1].interpolation(odl.ProductSpace(odl.rn(self.nb_pts_grid),2).element(grid),out=None,bounds_check=False)
        vect_field_interp=vect_field.space.element([A,B])
        for i, vi in enumerate(vect_field_interp):
            grid_velocity[i][:] = -vi.ntuple.asarray()
        return grid_velocity


    def apply_vect_field_adjoint_diff(self,mat):
        """from the matrix mat return the matrix corresponding to the
        adjoint for the action of vector fields on points of the grid"""
        return -mat


    def derivative(self,grid0):
        operator=self
        class Computederivative(Operator):
            def __init__(self,grid):
                self.grid=operator.domain.element(grid)
                super().__init__(domain=operator.domain, range=operator.range,linear=False)

            def _call(self,grid_der):
                grad = Gradient(domain=operator.range, method='central', pad_mode='symmetric')
                grad_templ = grad(operator.template)
                grad_templ_interp=operator.template.space.tangent_bundle.element([gf.interpolation(self.grid,out=None,bounds_check=False) for gf in grad_templ])
                return PointwiseInner(operator.template.space.tangent_bundle,grad_templ_interp )(grid_der)

            def adjoint(self,im):
                """ adjoint under the form of an element of domain but it is the representant in l2, ie a vector field"""
                grad = Gradient(domain=operator.range, method='central', pad_mode='symmetric')
                grad_templ = grad(operator.template)
                grad_templ_interp=operator.template.space.tangent_bundle.element([gf.interpolation(self.grid,out=None,bounds_check=False) for gf in grad_templ])
                grad_op=PointwiseInner(operator.template.space.tangent_bundle,grad_templ_interp ).adjoint(im)
                adj=operator.domain.zero()
                for i in range(0,2):
                    for j in range(0,operator.nb_pts_grid):
                        adj[i][j]=grad_op[i][j]
                return adj

        return Computederivative(grid0)



class obj_fun_dir(Functional):
        def __init__(self,data):
            self.data=data
            super().__init__(data.space,linear=False)

        def _call(self,template,out=None):
            im=self.data-template
            return im.inner(im)

        @property
        def gradient(self):
            functional=self
            class computegradient(Operator):
                def __init__(self):
                    super().__init__(domain=functional.domain, range=functional.domain,linear=False)

                def _call(self,X):
                    return 2*(X-functional.data)

            return computegradient()

#
#        """ derivate of the operator at ``grid`` """
#        print("deform image 72")
#        print(grid.space)
#        grid=self.domain.element(grid)
#        print("deform image 74")
#        grad = Gradient(domain=self.range, method='central',
#                        pad_mode='symmetric')
#        print("deform image 77")
#        grad_templ = grad(self.template)
#        print("deform image 79")
#        grad_templ_interp=self.template.space.tangent_bundle.element([gf.interpolation(grid,out=None,bounds_check=False) for gf in grad_templ])
#        #grad_templ_interp=self.domain.element(grad_templ_interp)
#        print("deform image 82")
#        print(grad_templ_interp.space)
#        return PointwiseInner(ProductSpace(),grad_templ_interp )
#

