#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 15:30:48 2017

@author: bgris
"""


"""Classes for large deformations defined thanks to control points."""

# Imports for common Python 2/3 codebase
#from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import super
import timeit
import numpy.matlib
from odl.solvers.functional.functional import Functional
import numpy as np
import odl
from odl.discr import DiscreteLp, Gradient, Divergence
from odl.operator import Operator, PointwiseInner
from odl.space import ProductSpace
from odl.space import rn
from odl.space.fspace import FunctionSpace
__all__ = ('LargeDeformFixedTemplCP','shooting_op','integrator','deform_op_CP')




class LargeDeformFixedTemplCP(Functional):

    """ Function which can compute a scalar energy (and its gradient) from
    a value of initial control points and momenta"""



    def __init__(self,obj_fun,apply_templ_op,deform_op_CP,metric, template):
        """Initialize a new instance.

        Parameters
        ----------
        obj_fun : data attachment term (input: image, output: scalar value)
        apply_templ_op : operator which has to be initialized with an image
                        and defines how is computed the new image,
                        deformed by a deformation
        deform_op_CP : deformation operator (input: CP and MOM, output:
            deformation), it must have a method apply_defo
        metric : metric for the space of generated vector fields (ex : RKHS)
        template : `DiscreteLpElement`

        """
        # TODO: check that spaces of template and deform_op are the same
        self.obj_fun= obj_fun
        self.template=template
        self.apply_templ_op=apply_templ_op
        self.apply_fixed_templ=self.apply_templ_op(self.template)
        self.deform_op_CP=deform_op_CP
        self.metric=metric

        super().__init__(deform_op_CP.domain,
                         linear=False)


    def _call(self,X,out=None):
#        apply_templ=self.apply_templ_op(self.deform_op.apply_defo(X),self.template)([1])
#        energy = self.obj_fun(apply_templ) + self.metric(X)
        energy = self.obj_fun(self.apply_fixed_templ.deform_from_operator(self.deform_op_CP.apply_defo(X)))
#        print("energy={}".format(energy))

        energy+= self.metric(X)
        print("energy={}".format(energy))
        return energy

    @property
    def gradient(self):
        functional=self
        class ComputeGradient(Operator):

            def __init__(self):
                self.spaceX=functional.deform_op_CP.domain
                self.spacegrid=functional.deform_op_CP.range
                self.space=odl.ProductSpace(self.spaceX,self.spacegrid)
                # the adjoint space is assimilated to the space
                self.spaceAdjoint=odl.ProductSpace(self.spaceX,self.spacegrid)
                self.spaceAdjointout=self.spaceX
                super().__init__(domain=self.spaceX,range=self.spaceAdjointout,linear=False)


            def _call(self,X0):
#                CP=X[0]
#                MOM=X[1]
       #         print('000000000000000000000000000000000')
#                start0 = timeit.default_timer()
                temp_list=functional.deform_op_CP.shooting_op.temporal_list_vect(X0,
                                                functional.template.space.tangent_bundle)
                X_temporal=temp_list[0]
#                CP_temporal=X_temporal[0]
#                MOM_temporal=X_temporal[1]
                VectField_temporal=temp_list[1]
          #      print('01111111111111111111111111')
                nb_time_points=functional.deform_op_CP.nb_time_points
            #    print('001111111111111111111111111')
                grid_defo=functional.apply_fixed_templ.get_used_grid(functional.deform_op_CP.apply_defo(X0))

           #     print('0001111111111111111111111111')
                Adjoint=self.spaceAdjoint.zero()

#                print('1111111111111111111111111')
#                print(grid_defo.space)
                #TODO: implementer le gradient pour mettre en Adjoint[1] la ligne suivante
#                B=functional.template.space.tangent_bundle.element([functional.template.space.element(gf) for gf in grid_defo])
#                print(B.space)
                A=(functional.obj_fun*functional.apply_fixed_templ).gradient(grid_defo)
#                print('2222222222222222222222222')
                """ Adjoint[1] under the form of an element of domain but it is the representant in l2, ie a vector field"""
                Adjoint[1]=A.copy()

#                print('333333333333333333333333333333')
#                print(A)

                time_step=functional.deform_op_CP.time_step
                grid_t=grid_defo
            #    print('222222222222222222222222222222')
#                end0 = timeit.default_timer()
#                print("time 0000000000000       {}".format(end0 - start0))
                for t in range(0,nb_time_points-1):
#                    print('aaa t={}'.format(t))
#                    print(Adjoint[0])
#                    start1 = timeit.default_timer()
                    d_Adjoint=self.spaceAdjoint.zero()
                    X=X_temporal[nb_time_points-t-1].copy()
#                    print('bbbb t={}'.format(t))
#                    print('aaa d_Adjoint t={}'.format(t))



                    """New version : begin """
#                    delta=0.0000001
#                    X_depl=X+delta*self.spaceX.element([-Adjoint[0][1],Adjoint[0][0]])
#                    Y=odl.ProductSpace(grid_defo.space,2).zero()
#                    Y[0]=grid_t
#                    Y_depl=Y
#                    Y_depl[1]+=delta*Adjoint[1]
#
#                    # we multiply by -1 because the grid is transported backward
#                    Y_depl[1]*=-1
#
#
#          #          print('ccc t={}'.format(t))
#                    # compute_speed_silent returns geodesic speed of X=(CP,MOM)
#                    # and of Y[1]=momentum of the grid
#                    speed=functional.deform_op_CP.shooting_op.compute_speed_silent(X,Y)
#                    speed_depl=functional.deform_op_CP.shooting_op.compute_speed_silent(X_depl,Y_depl)
#                    """we use the formula of Sylvain Arguillere
#                    giving the adjoint thanks to a directional derivative.
#                    We use the approximation of the directional derivative
#                    of the gradient of the hamiltonian, and for the gradient
#                    we use the fact that speeds are the sympletic gradients of
#                    the hamiltonian"""
#                    d_Adjoint[0][0]=(speed[0][1]-speed_depl[0][1])/delta
#                    d_Adjoint[0][1]=(speed_depl[0][0]-speed[0][0])/delta
#                    d_Adjoint[1]=(speed[1]-speed_depl[1])/delta

                    """New version : end """


                    """Old version : begin """


                    """ first adjoint of the action of X on itself"""
                    delta=0.0000001
                    X_depl=X+delta*self.spaceX.element([-Adjoint[0][1],Adjoint[0][0]])
          #          print('ccc t={}'.format(t))
                    speedX=functional.deform_op_CP.shooting_op.compute_speed(X)
                    speedX_depl=functional.deform_op_CP.shooting_op.compute_speed(X_depl)
                    """we use the formula of Sylvain Arguillere
                    giving the adjoint thanks to a directional derivative.
                    We use the approximation of the directional derivative
                    of the gradient of the hamiltonian, and for the gradient
                    we use the fact that speeds are the sympletic gradients of
                    the hamiltonian"""
                    d_Adjoint[0][0]=(speedX[1]-speedX_depl[1])/delta
                    d_Adjoint[0][1]=(speedX_depl[0]-speedX[0])/delta
#                    print('00000000000000000000000000000000')
##                    print(speedX_depl[0])
##                    print(speedX[0])
#                    print(d_Adjoint[0])
#                    end1 = timeit.default_timer()
#                    print("time 111111111111111111111111111111       {}".format(end1 - start1))

#                    print('ddd t={}'.format(t))
                    """second adjoint of the derivative with respect to X
                    of the action of X on the grid """
#                    start2 = timeit.default_timer()
                    vect_field_list=functional.deform_op_CP.shooting_op.derivate_vector_field_X(X,functional.template.space.tangent_bundle)
#                    print("vector field list 1 0")
#                    print(vect_field_list[1][0][0])
#                    print("vector field list 1 1")
#                    print(vect_field_list[1][0][1])
#                    end2 = timeit.default_timer()
#                    print("time 2222222222222222222222222222222222    {}".format(end2 - start2))
                    """adjoint[1] as a l2 element so that the inner product is the good one"""
#                    start23 = timeit.default_timer()
#                    print('00000000000000000000000000000000')
                    Adjoint_L2=functional.template.space.tangent_bundle.element(Adjoint[1])
#                    print('111111111111111111111111111111111111111111')

                    """ first loop on i1 so that it can be parallelized ?"""
                    for i1 in range(0,vect_field_list[0].size):
                        for i0 in range(0,vect_field_list.size):
                            for i2 in range(0,vect_field_list[i0][i1].size):
#                                print("i0={} i1={} i2={}".format(i0,i1,i2))
#                                start230 = timeit.default_timer()
                                app_grid=functional.apply_fixed_templ.apply_vect_field(vect_field_list[i0][i1][i2],grid_t)
#                                end230 = timeit.default_timer()
#                                print("time 22222222222222333333333333333333333333300000000000000000000000    {}".format(end230 - start230))
#                                print('22222222222222222222222222222222222222222222')
                                d_Adjoint[0][i0][i1][i2]+=Adjoint_L2.inner(functional.template.space.tangent_bundle.element(app_grid))
#                                print(Adjoint_L2.inner(functional.template.space.tangent_bundle.element(app_grid)))
#                                print(app_grid.inner(Adjoint[1]))
#                                print('33333333333333333333333333333333333333')
#                                start231 = timeit.default_timer()
#                                d_Adjoint[0][i0][i1][i2]+=app_grid.inner(Adjoint[1])
#                                end231 = timeit.default_timer()
#                                print("time 222222222222223333333333333333333333333111111111111111111111111    {}".format(end231 - start231))
#                    print('22222222222222222222222222222222222')
#                    print(d_Adjoint[0])





#                    end23 = timeit.default_timer()
#                    print("time 222222222222223333333333333333333333333    {}".format(end23 - start23))


                    """third adjoint of the derivative with respect to the grid
                    of the action of X on the grid """
#                    start3 = timeit.default_timer()
                    for d in range(0,grid_t.size):
                        Point=[grid_t[0][d],grid_t[1][d]]
                        mat=functional.apply_fixed_templ.apply_vect_field_adjoint_diff(functional.deform_op_CP.shooting_op.vect_field_adjoint_diff(X,Point))
                        d_Adjoint[1][0][d]+=mat[0][0]*Adjoint[1][0][d]+mat[0][1]*Adjoint[1][1][d]
                        d_Adjoint[1][1][d]+=mat[1][0]*Adjoint[1][0][d]+mat[1][1]*Adjoint[1][1][d]






#                    print(d_Adjoint[0])
#                    end3 = timeit.default_timer()
#                    print("time 33333333333333333333333333333333              {}".format(end3 - start3))
                    """Old version : end """
                    speed_grid=functional.apply_fixed_templ.apply_vect_field(VectField_temporal[nb_time_points-t-1],grid_t)
                    grid_t-=time_step*speed_grid
                    Adjoint=Adjoint+time_step*d_Adjoint
##                print('3333333333333333333333333333333')
##                print(Adjoint[0])


                speed=functional.deform_op_CP.shooting_op.compute_speed(X0)
                Adjoint[0][0]=Adjoint[0][0]-speed[1]
                Adjoint[0][1]=Adjoint[0][1]+speed[0]
                return Adjoint[0]

        #        print('4444444444444444444444444444444444444')
        return ComputeGradient()












class shooting_op(Operator):
    """ shooting operator which computes geodesics from initial values of
    CP and MOM """


    def __init__(self,nb_time_points, time_step,scale,NbCP,dim):
        self.nb_time_points=nb_time_points
        self.time_step =time_step
        self.scale =scale
        self.NbCP =NbCP
        self.dim=dim
        domain=odl.ProductSpace(odl.ProductSpace(odl.rn(dim), NbCP),2)

        super().__init__(domain=domain,range=odl.ProductSpace(odl.ProductSpace(odl.ProductSpace(odl.rn(dim),NbCP),2),nb_time_points),linear=False)


    def _call(self,X):
        Y=self.range.element()
        Y[0]=X
        for t in range(0,self.nb_time_points-1):
            U=self.compute_speed(Y[t])
            Y[t+1]=Y[t]+self.time_step*U

        return Y


    def compute_speed(self,X):
        U=self.domain.zero()
        CP=X[0]
        MOM=X[1]

        for i in range(self.NbCP):
            for j in range(self.NbCP):
                expo=np.exp(-((CP[i][0]-CP[j][0]) ** 2 + (CP[i][1]-CP[j][1])  ** 2) / (2 * self.scale ** 2))
                U[0][i]=U[0][i]+expo*MOM[j]
                U[1][i]=U[1][i]-(-1/(self.scale**2))*(MOM[i][0]*MOM[j][0]+MOM[i][1]*MOM[j][1])*expo*(CP[i]-CP[j])

        return U

    def compute_speed_silent(self,X,Y):
        # compute_speed_silent returns geodesic speed of X with X[1] the
        # momentum of X[0] and of Y[1]=momentum of Y[0] in the case where there
        # is a silent variable Y
        # Be careful : X is in ProductSpace(ProductSpace(rn(2), NbCP), 2) but
        # Y is in ProductSpace(ProductSpace(rn(NbPtsGrid), 2), 2)
        U=self.domain.zero()
        V=Y[0].space.zero()
        NbPtsGrid=Y[0][0].space.size
        CP=X[0]
#        MOM=X[1]

        # contains K(CP_i, CP_j)
        K_CPCP=np.zeros((self.NbCP,self.NbCP))
        # contains K(CP_i, grid_j)
        K_CPgr=np.zeros((self.NbCP,NbPtsGrid))

        for i in range(self.NbCP):
            for j in range(self.NbCP):
                K_CPCP[i][j]=np.exp(-((CP[i][0]-CP[j][0]) ** 2 + (CP[i][1]-CP[j][1])  ** 2) / (2 * self.scale ** 2))

            for k in range(NbPtsGrid):
                K_CPgr[i][k]=np.exp(-((CP[i][0]-Y[0][0][k]) ** 2 + (CP[i][1]-Y[0][1][k])  ** 2) / (2 * self.scale ** 2))

        convolve_MOMgrid=np.dot(K_CPgr,np.transpose(Y[1]))
#        convolve_MOMCP=np.dot(K_CPCP,X[1])

        # controls
        alpha = X[1] + np.linalg.solve(K_CPCP,convolve_MOMgrid)

        U[0]= np.dot(K_CPCP,alpha)

        CP_rep0=numpy.matlib.repmat(np.transpose(CP)[0],self.NbCP,1)
        CP_rep1=numpy.matlib.repmat(np.transpose(CP)[1],self.NbCP,1)

        # arrays whose comp (i,j) is K(CP_i,CP_j)*CP(j)
        multKCP0=numpy.multiply(K_CPCP,CP_rep0)
        multKCP1=numpy.multiply(K_CPCP,CP_rep1)

        A0=np.dot(np.transpose(multKCP0),alpha)
        A1=np.dot(np.transpose(multKCP1),alpha)

        # equal to U[1].transpose
        W=np.transpose(U[1])
        W[0]+=np.sum(np.multiply(X[1],A0),1)
        W[1]+=np.sum(np.multiply(X[1],A1),1)

        B0=np.dot(multKCP0,alpha)
        B1=np.dot(multKCP1,alpha)

        W[0]-=np.sum(np.multiply(X[1],B0),1)
        W[1]-=np.sum(np.multiply(X[1],B1),1)


        CP_repgr0=np.transpose(numpy.matlib.repmat(np.transpose(CP)[0],NbPtsGrid,1))
        CP_repgr1=np.transpose(numpy.matlib.repmat(np.transpose(CP)[1],NbPtsGrid,1))
        gr_rep0=numpy.matlib.repmat(Y[0][0],self.NbCP,1)
        gr_rep1=numpy.matlib.repmat(Y[0][1],self.NbCP,1)

        # arrays whose comp (i,j) is K(CP_i,gr_j)*CP(i)
        multKCPgr0=numpy.multiply(K_CPgr,CP_repgr0)
        multKCPgr1=numpy.multiply(K_CPgr,CP_repgr1)
        # arrays whose comp (i,j) is K(CP_i,gr_j)*gr(j)
        multKgr0=numpy.multiply(K_CPgr,gr_rep0)
        multKgr1=numpy.multiply(K_CPgr,gr_rep1)

        C0=np.dot(multKCPgr0,np.transpose(Y[1]))
        C1=np.dot(multKCPgr1,np.transpose(Y[1]))
        W[0]+=np.sum(np.multiply(alpha,C0),1)
        W[1]+=np.sum(np.multiply(alpha,C1),1)

        D0=np.dot(multKgr0,np.transpose(Y[1]))
        D1=np.dot(multKgr1,np.transpose(Y[1]))
        W[0]-=np.sum(np.multiply(alpha,D0),1)
        W[1]-=np.sum(np.multiply(alpha,D1),1)



        E0=np.dot(np.transpose(multKgr0),alpha)
        E1=np.dot(np.transpose(multKgr1),alpha)
        V[0]+=rn(NbPtsGrid).element(np.transpose(np.sum(np.multiply(np.transpose(Y[1]),E0),1)).copy())
        V[1]+=rn(NbPtsGrid).element(np.transpose(np.sum(np.multiply(np.transpose(Y[1]),E1),1)).copy())

        F0=np.dot(np.transpose(multKCPgr0),alpha)
        F1=np.dot(np.transpose(multKCPgr1),alpha)
        V[0]-=rn(NbPtsGrid).element(np.transpose(np.sum(np.multiply(np.transpose(Y[1]),F0),1)).copy())
        V[1]-=rn(NbPtsGrid).element(np.transpose(np.sum(np.multiply(np.transpose(Y[1]),F1),1)).copy())

        V=np.dot((-1/(self.scale ** 2)),V)
        W=np.dot((-1/(self.scale ** 2)),W)
        U[1]=np.transpose(W)
        return [U,V]


    def temporal_list_vect(self,X,disp_field_space):

        # disp_field_space.element() defines a vector field
        Y=self.range.element()
        Z=odl.ProductSpace(disp_field_space,self.nb_time_points).element()
        Y[0]=X
        CP=X[0]
        MOM=X[1]
        for t in range(0,self.nb_time_points):
            U=self.compute_speed(Y[t])
            if(t<self.nb_time_points -1):
                Y[t+1]=Y[t]+self.time_step*U
            def VX(x):
                a=0
                for k in range(0,self.NbCP):
                    a=a+MOM[k][0]* np.exp(-((x[0]-CP[k][0]) ** 2 + (x[1]-CP[k][1])  ** 2) / (2 * self.scale ** 2))
                return a

            def VY(x):
                a=0
                for k in range(0,self.NbCP):
                    a=a+MOM[k][1]* np.exp(-((x[0]-CP[k][0]) ** 2 + (x[1]-CP[k][1])  ** 2) / (2 * self.scale ** 2))
                return a

            disp_func = [VX,VY]

            Z[t] = disp_field_space.element(disp_func)

        return [Y,Z]


    def derivate_vector_field_X(self,X,disp_field_space):
        derivate_list=odl.ProductSpace(odl.ProductSpace(odl.ProductSpace(disp_field_space,self.dim), self.NbCP),2).element()
        CP=X[0]
        MOM=X[1]

        for k in range(0,self.NbCP):
                def VX(x):
                    a=MOM[k][0]*(x[0]-CP[k][0])*(1/self.scale ** 2)* np.exp(-((x[0]-CP[k][0]) ** 2 + (x[1]-CP[k][1])  ** 2) / (2 * self.scale ** 2))
                    return a

                def VY(x):
                    a=MOM[k][1]*(x[0]-CP[k][0])*(1/self.scale ** 2)* np.exp(-((x[0]-CP[k][0]) ** 2 + (x[1]-CP[k][1])  ** 2) / (2 * self.scale ** 2))
                    return a

                disp_func = [VX,VY]

                derivate_list[0][k][0] = disp_field_space.element(disp_func)

                def VX(x):
                    a=MOM[k][0]*(x[1]-CP[k][1])*(1/self.scale ** 2)* np.exp(-((x[0]-CP[k][0]) ** 2 + (x[1]-CP[k][1])  ** 2) / (2 * self.scale ** 2))
                    return a

                def VY(x):
                    a=MOM[k][1]*(x[1]-CP[k][1])*(1/self.scale ** 2)* np.exp(-((x[0]-CP[k][0]) ** 2 + (x[1]-CP[k][1])  ** 2) / (2 * self.scale ** 2))
                    return a

                disp_func = [VX,VY]

                derivate_list[0][k][1] = disp_field_space.element(disp_func)


                def VX(x):
                    a=np.exp(-((x[0]-CP[k][0]) ** 2 + (x[1]-CP[k][1])  ** 2) / (2 * self.scale ** 2))
                    return a

                def VY(x):
                    a=0
                    return a

                disp_func = [VX,VY]

                derivate_list[1][k][0] = disp_field_space.element(disp_func)

                def VX(x):
                    a=0
                    return a

                def VY(x):
                    a=np.exp(-((x[0]-CP[k][0]) ** 2 + (x[1]-CP[k][1])  ** 2) / (2 * self.scale ** 2))
                    return a

                disp_func = [VX,VY]

                derivate_list[1][k][1] = disp_field_space.element(disp_func)

        return derivate_list

    def vect_field_adjoint_diff(self,X,Point):
        """ returns the transpose of the differential of the
        generated vector field at Point"""
        mat=odl.ProductSpace(odl.rn(self.dim),self.dim).zero()
        """mat[u][v] equals the derivative of the v-th component of the generated
        vector field with respect to the u-th direction"""
        CP=X[0]
        MOM=X[1]
        for k in range(0,self.NbCP):
            a=np.exp(-((Point[0]-CP[k][0]) ** 2 + (Point[1]-CP[k][1])  ** 2) / (2 * self.scale ** 2))
            for u0 in range(0,self.dim):
                for u1 in range(0,self.dim):
                    mat[u0][u1]-=MOM[k][u1]*a*(Point[u0]-CP[k][u0])*(1/self.scale ** 2)

        return mat









class integrator(Operator):
    """ integrator which takes in input a list of vector field and return
    the corresponding large deformation (deformation of the point of the grid
    on which vector fields are defined)"""


    def __init__(self,vector_fields_list,time_step):
        self.vector_fields_list=vector_fields_list
        self.nb_time_points=vector_fields_list.size
        self.nb_pts_grid=vector_fields_list[0][0].shape[0]*vector_fields_list[0][0].shape[1]
        self.time_step=time_step

        super().__init__(domain=odl.rn(1),range=odl.ProductSpace(odl.rn(self.nb_pts_grid),2),linear=False)

    def _call(self,j):
        grid_points=self.vector_fields_list[0].space[0].points()
#        if (j>self.nb_time_points):
#            j=self.nb_time_points

        for t in range (0,self.nb_time_points):
            for i, vi in enumerate(self.vector_fields_list[t]):
                grid_points[:, i] += self.time_step*vi.ntuple.asarray()

        return self.range.element(grid_points.T)


    def temporal_list_grid(self):
#        Y=self.range.element()
        Z=odl.ProductSpace(self.range,self.nb_time_points-1).element()
        Z[0]=self.vector_fields_list[0].space[0].points()
#        grid_points=self.vector_fields_list[0].space[0].points()
#        if (j>self.nb_time_points):
#            j=self.nb_time_points

        for t in range (0,self.nb_time_points-1):
            for i, vi in enumerate(self.vector_fields_list[t]):
                Z[t+1][:, i] =Z[t][:, i] + self.time_step*vi.ntuple.asarray()

        return Z



    def inverse(self):
        grid_points=self.vector_fields_list[0].space[0].points()
        for t in range (0,self.nb_time_points):
            for i, vi in enumerate(self.vector_fields_list[self.nb_time_points-t-1]):
                grid_points[:, i] -= self.time_step*vi.ntuple.asarray()

        return self.range.element(grid_points.T)



class deform_op_CP(Operator):

    def __init__(self,scale,NbCP, space,nb_time_points, time_step, domain=None):
        self.space=space
        self.ndim=self.space.ndim
        self.scale=scale
        self.NbCP=NbCP
        self.nb_time_points=nb_time_points
        self.time_step=time_step
        self.nb_pts_grid=space.shape[0]*space.shape[1]
        self.shooting_op=shooting_op(nb_time_points, time_step,scale,NbCP,self.ndim)
        domain=odl.ProductSpace(odl.ProductSpace(odl.rn(self.ndim), NbCP),2)
        super().__init__(domain=domain,range=odl.ProductSpace(odl.rn(self.nb_pts_grid),2),linear=False)

    def _call(self,X):
        L=self.shooting_op.temporal_list_vect(X,self.space.tangent_bundle)
        integrator_op=integrator(L[1],self.time_step)
        return integrator_op([0])


    def inverse(self,X):
        L=self.shooting_op.temporal_list_vect(X,self.space.tangent_bundle)
        integrator_op=integrator(L[1],self.time_step)
        return integrator_op.inverse()


    def apply_defo(self,X):
        operator = self
        class apply_defo_op(Operator):
            def __init__(self,Y):
                self.X0=Y
                self.nb_pts_grid=operator.nb_pts_grid
                super().__init__(domain=odl.rn(1),range=odl.ProductSpace(odl.rn(self.nb_pts_grid),2),linear=False)

            def _call(self,j):
                L=self.shooting_op.temporal_list_vect(self.X0,operator.space.tangent_bundle)
                integrator_op=integrator(L[1],operator.time_step)
                return integrator_op(operator.nb_time_points)

            def inverse(self):
                L=operator.shooting_op.temporal_list_vect(self.X0,operator.space.tangent_bundle)
                integrator_op=integrator(L[1],operator.time_step)
                return integrator_op.inverse()
        return apply_defo_op(X)




















