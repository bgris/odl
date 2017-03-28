#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 09:49:56 2017

@author: bgris
"""




import numpy as np
import odl
import numpy

#%%
templ_space = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20], shape=[100, 100])
template = odl.phantom.shepp_logan(templ_space, modified=True)
disp_field_space = templ_space.tangent_bundle

nb_time_points=11
time_step=0.1
NbCP=1
scale=10
dim=2
CPspace=odl.ProductSpace(odl.rn(template.space.ndim), NbCP)
MOMspace=odl.ProductSpace(odl.rn(template.space.ndim), NbCP)
SpaceTot=odl.ProductSpace(CPspace,MOMspace)

shoot_op=odl.deform.shooting_op(nb_time_points, time_step,scale,NbCP,dim)
CP=CPspace.element([[0,0]])
MOM=MOMspace.element([[8,0]])
X0=SpaceTot.element([CP,MOM])

#
deform_op=odl.deform.deform_op_CP(scale,NbCP, templ_space,nb_time_points, time_step, domain=None)
#grid_defo=deform_op.inverse(X)
#template_def=template.interpolation(grid_defo,out=None,bounds_check=False)
#template_def=templ_space.element(template_def)
#template_def.show();
#
#



apply_defo=deform_op.apply_defo(X0)
grid=apply_defo.inverse()


apply_templ_op=odl.deform.DeformTemplateGeometric
apply_templ=apply_templ_op(template)
template_defo=apply_templ.deform_from_operator(deform_op.apply_defo(X0))
template.show();
template_defo.show();

angle_partition = odl.uniform_partition(0, np.pi, 180)
detector_partition = odl.uniform_partition(-20, 20, 128)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)
#ray_trafo = odl.tomo.RayTransform(templ_space, geometry,impl='astra_cpu')
ray_trafo = odl.tomo.RayTransform(templ_space, geometry)
deformed_template_true=apply_templ.deform_from_operator(deform_op.apply_defo(X0))
data = ray_trafo(deformed_template_true)
obj_fun = odl.solvers.L2NormSquared(ray_trafo.range) * (ray_trafo - data)
obj_fun_dir = odl.deform.obj_fun_dir(deformed_template_true)




metric=odl.deform.RKHSNorm(scale,NbCP,SpaceTot,domain=None)

functional=odl.deform.LargeDeformFixedTemplCP(5000*obj_fun_dir,apply_templ_op,deform_op,metric, template)

CP_test=CPspace.element([[-5,-5]])
MOM_test=MOMspace.element([[-5,-5]])
X_test=SpaceTot.element([CP_test,MOM_test])

nb_time_points=3
time_step=0.5

scale_test=5

Xmax=round(24/(1*scale_test))
Ymax=round(30/(1*scale_test))
Xlin=numpy.linspace(-12,12,Xmax)
Ylin=numpy.linspace(-15,15,Ymax)
CP_test=[]
MOM_test=[]
NbCP_test=0
for i in range(0,Xmax):
    for j in range(0,Ymax):
        CP_test.append([Xlin[i],Ylin[j]])
        MOM_test.append([0,0])
        NbCP_test=NbCP_test+1


CPspace_test=odl.ProductSpace(odl.rn(template.space.ndim), NbCP_test)
MOMspace_test=odl.ProductSpace(odl.rn(template.space.ndim), NbCP_test)
SpaceTot_test=odl.ProductSpace(CPspace_test,MOMspace_test)
CP_test=CPspace_test.element(CP_test)
MOM_test=MOMspace_test.element(MOM_test)
X_test=SpaceTot_test.element([CP_test,MOM_test])

shoot_op_test=odl.deform.shooting_op(nb_time_points, time_step,scale_test,NbCP_test,dim)
deform_op_test=odl.deform.deform_op_CP(scale_test,NbCP_test, templ_space,nb_time_points, time_step, domain=None)
apply_templ_op_test=odl.deform.DeformTemplateGeometric
apply_templ_test=apply_templ_op_test(template)
metric_test=odl.deform.RKHSNorm(scale_test,NbCP_test,SpaceTot_test,domain=None)

functional_test=odl.deform.LargeDeformFixedTemplCP(5000*obj_fun,apply_templ_op_test,deform_op_test,metric_test, template)

line_search=odl.solvers.BacktrackingLineSearch(functional_test, tau=0.8, discount=0.001, alpha=0.0010,
                 max_num_iter=60, estimate_step=True)
X=X_test.copy()
SpaceTot_grid=odl.ProductSpace(SpaceTot,grid.space)
#
#%%
energy_init =functional_test(X)
print('Initial energy {}'.format(energy_init))

A=odl.solvers.steepest_descent(functional_test, X,line_search, maxiter=3, tol=1e-16,
                     projection=None, callback=None)



import timeit

start = timeit.default_timer()
gr1=functional_test.gradient(X)
end = timeit.default_timer()
print(end - start)


#%%

import timeit
A=[6,11,21,41,61,101]
B=odl.ProductSpace(odl.ProductSpace(odl.rn(10000), 2),6).element()
C=odl.ProductSpace(odl.rn(1),4).element()
for i in range(4,5):
    print(i)
    nb_time_points=A[i]
    time_step=1/(nb_time_points-1)

    shoot_op_test=odl.deform.shooting_op(nb_time_points, time_step,scale_test,NbCP_test,dim)
    deform_op_test=odl.deform.deform_op_CP(scale_test,NbCP_test, templ_space,nb_time_points, time_step, domain=None)
    functional_test=odl.deform.LargeDeformFixedTemplCP(5000*obj_fun,apply_templ_op_test,deform_op_test,metric_test, template)

    start = timeit.default_timer()
    gr1=deform_op_test(X)
    gr2=functional_test(X)
    end = timeit.default_timer()
    B[i]=gr1
    C[i]=gr2
    print(end - start)
    template_defo=apply_templ.deform_from_operator(deform_op_test.apply_defo(X0))
    template_defo.show(i)




#%%

import timeit
A=[6,11,21,41,61]
B=odl.ProductSpace(X.space,4).element()
for i in range(0,3):
    print(i)
    nb_time_points=A[i]
    time_step=1/(nb_time_points-1)

    shoot_op_test=odl.deform.shooting_op(nb_time_points, time_step,scale_test,NbCP_test,dim)
    deform_op_test=odl.deform.deform_op_CP(scale_test,NbCP_test, templ_space,nb_time_points, time_step, domain=None)
    functional_test=odl.deform.LargeDeformFixedTemplCP(5000*obj_fun,apply_templ_op_test,deform_op_test,metric_test, template)

    start = timeit.default_timer()
    gr1=functional_test.gradient(X)
    end = timeit.default_timer()
    B[i]=gr1
    print(end - start)

#%%
delta=0.000001

for i in range(0,3):
    X_bis=X-delta*B[i]
    nb_time_points=A[i]
    time_step=1/(nb_time_points-1)

    shoot_op_test=odl.deform.shooting_op(nb_time_points, time_step,scale_test,NbCP_test,dim)
    deform_op_test=odl.deform.deform_op_CP(scale_test,NbCP_test, templ_space,nb_time_points, time_step, domain=None)
    functional_test=odl.deform.LargeDeformFixedTemplCP(5000*obj_fun,apply_templ_op_test,deform_op_test,metric_test, template)

    print(functional_test(X))
    template_defo_testi=apply_templ.deform_from_operator(deform_op_test.apply_defo(X_bis))
    template_defo_testi.show("{}".format(i))


