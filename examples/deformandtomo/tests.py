#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 14:13:10 2017

@author: bgris
"""




import numpy as np
import odl
import numpy

#%%
templ_space = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20], shape=[50, 50])
template = odl.phantom.shepp_logan(templ_space, modified=True)
disp_field_space = templ_space.tangent_bundle

#nb_time_points=11
#time_step=0.1
#NbCP=2
#scale=3
#dim=2
#CPspace=odl.ProductSpace(odl.rn(template.space.ndim), NbCP)
#MOMspace=odl.ProductSpace(odl.rn(template.space.ndim), NbCP)
#SpaceTot=odl.ProductSpace(CPspace,MOMspace)
#shoot_op=odl.deform.shooting_op(nb_time_points, time_step,scale,NbCP,dim)
#
#CP=CPspace.element([[-10,-10],[10,10]])
#MOM=MOMspace.element([[0,10],[0,0]])
#X=SpaceTot.element([CP,MOM])

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
#nb_time_points=11
#time_step=0.1
#NbCP=3
#scale=3
#dim=2
#CPspace=odl.ProductSpace(odl.rn(template.space.ndim), NbCP)
#MOMspace=odl.ProductSpace(odl.rn(template.space.ndim), NbCP)
#SpaceTot=odl.ProductSpace(CPspace,MOMspace)
#shoot_op=odl.deform.shooting_op(nb_time_points, time_step,scale,NbCP,dim)
#
#CP=CPspace.element([[-10,-10],[-10,10],[10,-10]])
#MOM=MOMspace.element([[0,10],[0,0],[0,0]])
#X=SpaceTot.element([CP,MOM])


#L=shoot_op(X)
#L=shoot_op.temporal_list_vect(X,disp_field_space)
#
#integrator_op=odl.deform.integrator(L[1],time_step)
#grid_defo=integrator_op([1])

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

##%%
##
#
#f=apply_templ.derivative(grid)
#f(grid)
#f.adjoint(template)
#
#range=template.space
#grid_def=odl.space.ProductSpace(odl.space.rn(400), 2).element(grid)
#from odl.discr import DiscreteLp, Gradient, Divergence
#grad = Gradient(domain=range, method='central', pad_mode='symmetric')
#from odl.operator import Operator, PointwiseInner
#grad_templ = grad(template)
#def_grad=template.space.tangent_bundle.element([gf.interpolation(grid,out=None,bounds_check=False).T for gf in grad_templ])
#der=PointwiseInner(odl.ProductSpace(odl.rn(400),2),def_grad)(grid_def)
#b=apply_templ(grid)
#a=apply_templ.derivative(grid)(grid)
#

#
##%%
#
#vect_field_diff=apply_templ.apply_vect_field_adjoint_diff(shoot_op.vect_field_adjoint_diff(X,X[0][1]))
#vect_field_list=shoot_op.derivate_vector_field_X(X,templ_space.tangent_bundle)
#grid_defo=apply_templ.apply_vect_field(vect_field_list[0][0][0],grid)
#
##%%
#vect_field=vect_field_list[0][0][0]
#nb_pts_grid=template.space.shape[0]*template.space.shape[1]
#domain=odl.ProductSpace(odl.rn(nb_pts_grid),2)
#grid_velocity=domain.element()
#A=vect_field[0].interpolation(odl.ProductSpace(odl.rn(nb_pts_grid),2).element(grid),out=None,bounds_check=False)
#B=vect_field[1].interpolation(odl.ProductSpace(odl.rn(nb_pts_grid),2).element(grid),out=None,bounds_check=False)
#vect_field_interp=vect_field.space.element([A,B])
#for i, vi in enumerate(vect_field_interp):
#    grid_velocity[i] = -vi.ntuple.asarray()
##%%
#
#vect_field_list=shoot_op.temporal_list_vect(X,templ_space.tangent_bundle)
#vect_field=vect_field_list[1][0]
#A=vect_field[0].interpolation(grid_defo,out=None,bounds_check=False)
#B=vect_field[1].interpolation(grid_defo,out=None,bounds_check=False)
#vect_field_interp=disp_field_space.element([A,B])
#


#%%

angle_partition = odl.uniform_partition(0, np.pi, 180)
detector_partition = odl.uniform_partition(-20, 20, 128)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)
ray_trafo = odl.tomo.RayTransform(templ_space, geometry)
deformed_template_true=apply_templ.deform_from_operator(deform_op.apply_defo(X0))
data = ray_trafo(deformed_template_true)
obj_fun = odl.solvers.L2NormSquared(ray_trafo.range) * (ray_trafo - data)
obj_fun_dir = odl.deform.obj_fun_dir(deformed_template_true)




metric=odl.deform.RKHSNorm(scale,NbCP,SpaceTot,domain=None)

functional=odl.deform.LargeDeformFixedTemplCP(5000*obj_fun_dir,apply_templ_op,deform_op,metric, template)
#%%
#functional(X0)
#
#grad=functional.gradient
#X=X0.copy()
##gr=grad(X)
#X[1][0][0]=-5.0
#X[1][0][1]=-5.0
#
#gr=grad(X)
#print("grad = {}".format(gr))
##%%
##CP=CPspace.element([[-10,-10]])
##MOM=MOMspace.element([[-10,20]])
##X=SpaceTot.element([CP,MOM])
#
#h=0.000000001
#energie=functional(X)
#X_test=X.copy();
#X_test[1][0][0]=X[1][0][0]+h
#energie_test=functional(X_test)
#met=metric(X)
#met_test=metric(X_test)
#diffmet=(met_test - met)/h
#diff=(energie_test - energie)/h
#print(diffmet)
#print(diff)
##%%
#templ_space = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20], shape=[250, 250])
#template_def = 1*odl.phantom.shepp_logan(templ_space, modified=True)
#template_def.show();
#grid=template.space.points()
#grid_def=template.space.points()
#displacement=10*template.space.tangent_bundle.one()
#for i, vi in enumerate(displacement):
#    grid_def[:,i]-=vi.ntuple.asarray()
#
#template_def=template.space.element(template.interpolation(grid_def.T,out=None,bounds_check=False))
#template_def.show();
#h=0.1
#displacement_bis= (1+h)*displacement
#grid_def_bis=template.space.points()
#for i, vi in enumerate(displacement_bis):
#    grid_def_bis[:,i]-=vi.ntuple.asarray()
#template_def_bis=template.space.element(template.interpolation(grid_def_bis.T,out=None,bounds_check=False))
#template_def_bis.show();
#diff=template.space.element(template_def_bis - template_def)
#
#diff.show();
#print((obj_fun_dir(template_def_bis) - obj_fun_dir(template_def) )/h)
##%%
#grid=template.space.points().T
#grid0=odl.ProductSpace(odl.rn(2500), 2).element(grid)
#grid_defo=deform_op.inverse(X)
#A=(functional.obj_fun*functional.apply_fixed_templ).gradient(grid_defo)
#
#im_adj0=template.space.element(A[0])
#im_adj0.show();
#im_adj1=template.space.element(A[1])
#im_adj1.show();
#im_adj_tangent=template.space.tangent_bundle.element(A)
#im_adj_tangent.show();
##%%
#Adjoint=odl.ProductSpace(SpaceTot,deform_op.range).zero()
#A=(obj_fun*apply_templ).gradient(odl.ProductSpace(odl.rn(2500), 2).zero())
##A=(obj_fun*apply_templ).gradient(grid_defo)
#vect_field_list=deform_op.shooting_op.derivate_vector_field_X(X,disp_field_space)
#
#Adjoint[1]=A
#app_grid=apply_templ.apply_vect_field(vect_field_list[1][0][0],grid_defo)
#prod0=app_grid.inner(Adjoint[1])
#appr_grid_temp=template.space.element(app_grid)
##%%
#
#CP_test=CPspace.element([[-10,-10],[10,10]])
#MOM_test=MOMspace.element([[0,0],[0,0]])

CP_test=CPspace.element([[-5,-5]])
MOM_test=MOMspace.element([[-5,-5]])
X_test=SpaceTot.element([CP_test,MOM_test])

nb_time_points=21
time_step=0.05

scale_test=8

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



#X_test=SpaceTot_test.element([[[0,0]],[[0,0]]])

shoot_op_test=odl.deform.shooting_op(nb_time_points, time_step,scale_test,NbCP_test,dim)
deform_op_test=odl.deform.deform_op_CP(scale_test,NbCP_test, templ_space,nb_time_points, time_step, domain=None)
apply_templ_op_test=odl.deform.DeformTemplateGeometric
apply_templ_test=apply_templ_op_test(template)
metric_test=odl.deform.RKHSNorm(scale_test,NbCP_test,SpaceTot_test,domain=None)

functional_test=odl.deform.LargeDeformFixedTemplCP(5000*obj_fun,apply_templ_op_test,deform_op_test,metric_test, template)

#%%
line_search=odl.solvers.BacktrackingLineSearch(functional_test, tau=0.8, discount=0.001, alpha=0.0010,
                 max_num_iter=60, estimate_step=True)
X=X_test.copy()
#SpaceTot_grid=odl.ProductSpace(SpaceTot,grid.space)
#X=SpaceTot_grid.element([X,grid])

energy_init =functional_test(X)
print('Initial energy {}'.format(energy_init))

A=odl.solvers.steepest_descent(functional_test, X,line_search, maxiter=30, tol=1e-16,
                     projection=None, callback=None)


#%%
import cProfile as profile
profile.run('A=odl.solvers.steepest_descent(functional_test, X,line_search, maxiter=2, tol=1e-16,projection=None, callback=None)', sort='tottime')
#%%
import profile

profile.run('A=odl.solvers.steepest_descent(functional_test, X,line_search, maxiter=2, tol=1e-16,projection=None, callback=None)', sort='tottime')
#%%

profile.run('functional_test.gradient(X)', sort='tottime')

#%%
import timeit

start = timeit.default_timer()
functional_test.gradient(X)
end = timeit.default_timer()
print(end - start)

#%%

grad = odl.Gradient(template.space, method='central', pad_mode='symmetric')
grad_templ = grad(template)
grad_templ_interp=template.space.tangent_bundle.element([gf.interpolation(grid,out=None,bounds_check=False) for gf in grad_templ])

A=odl.PointwiseInner(template.space.tangent_bundle,grad_templ_interp )(grid)

nb_pts_grid=template.space.shape[0]*template.space.shape[1]
spacegrid=odl.ProductSpace(odl.rn(nb_pts_grid),2)

B=odl.PointwiseInner(spacegrid,grad_templ_interp )(grid)

#%%

A=range(10)
B=range(3)
C=range(5)
D=[A,B,C]

for i in D:
    print(i[10])

#%%
template.show('Initial template');
deformed_template_true.show('Deformed template ground truth');
deform_template_estimated=apply_templ.deform_from_operator(deform_op_test.apply_defo(X))
deform_template_estimated.show('Estimated deformed template ');
deform_template_diff=np.sqrt((deformed_template_true-template)**2)
deform_template_diff.show('Difference before optimisation');
deform_template_diff=np.sqrt((deformed_template_true-deform_template_estimated)**2)
deform_template_diff.show('Difference after optimisation');
#%%
data_template = ray_trafo(template)
data_est = ray_trafo(deform_template_estimated)
data_template.show(title='Projection template (sinogram)');
data_est.show(title='Projection estimated deformed data (sinogram)');
data.show(title='Projection deformed data (ground truth, sinogram)');
residual_data=np.sqrt((data_template-data)**2) #/(data+1)
residual_data.show('Initial differences differences');
residual_data=np.sqrt((data_est-data)**2 )#/(data+1)
residual_data.show('differences');

#%%

import matplotlib.pyplot as plt
plt.plot([1,2,3,4])
plt.quiver([0],[1],[2],[3])
plt.ylabel('some numbers')
plt.show()


#%%
NbCP_test=4
scale_test=10

CPspace_test=odl.ProductSpace(odl.rn(template.space.ndim), NbCP_test)
MOMspace_test=odl.ProductSpace(odl.rn(template.space.ndim), NbCP_test)
SpaceTot_test=odl.ProductSpace(CPspace_test,MOMspace_test)
CP_test=CPspace_test.element([[-5, 5],[5, 5],[-5, -5],[5, -5]])
MOM_test=0.9*MOMspace_test.element([[0,0],[0,0],[0,0],[0,0]])
X_test=SpaceTot_test.element([CP_test,MOM_test])
nb_time_points=21
time_step=0.05
deform_op_test=odl.deform.deform_op_CP(scale_test,NbCP_test, templ_space,nb_time_points, time_step, domain=None)

metric_test=odl.deform.RKHSNorm(scale_test,NbCP_test,SpaceTot_test,domain=None)

functional_test=odl.deform.LargeDeformFixedTemplCP(5000*obj_fun_dir,apply_templ_op_test,deform_op_test,metric_test, template)

functional_test(X_test)
gr=functional_test.gradient(X)

X_bis=X-0.000001*gr
deform_template_estimated=apply_templ.deform_from_operator(deform_op_test.apply_defo(X_bis))
deform_template_estimated.show('Estimated deformed template ');
functional_test(X)
functional_test(X_bis)

deform_template_estimated_prev=apply_templ.deform_from_operator(deform_op_test.apply_defo(X))
deform_template_estimated_prev.show('Estimated deformed template _prev');

deform_template_diff=np.sqrt((deform_template_estimated-deform_template_estimated_prev)**2)
deform_template_diff.show('Difference after optimisation');


template.show('Initial template');
#%%
energy_final=functional(X)

apply_templ.derivative(grid).adjoint(template)

h=obj_fun*apply_templ

grad_obj=obj_fun.gradient(apply_templ(grid))
adj=apply_templ.derivative(grid).adjoint
adj(grad_obj)
a=apply_templ.derivative(grid).adjoint(obj_fun.gradient(apply_templ(grid)))
h.gradient(grid)
#%%
# test : changement d'espace pour une image
grid_points=L[1][0].space[0].points()
for i, vi in enumerate(L[1][0]):
    grid_points[:, i] -= 0.5*vi.ntuple.asarray()

template_def=template.interpolation(odl.ProductSpace(odl.rn(40000),2).element(grid_points.T),out=None,bounds_check=False)
template_def=templ_space.element(template_def)
template_def.show()

#%%




templ_space = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20], shape=[200, 200])
template = odl.phantom.shepp_logan(templ_space, modified=True)

NbCP=2
scale=10
CPspace=odl.ProductSpace(odl.rn(template.space.ndim), NbCP)
MOMspace=odl.ProductSpace(odl.rn(template.space.ndim), NbCP)
SpaceTot=odl.ProductSpace(CPspace,MOMspace)

CP=CPspace.element([[0,0],[10,10]])
MOM=MOMspace.element([[0,10],[5,0]])
X=SpaceTot.element([CP,MOM])

vector_field_op=odl.deform.CPVectorField(scale,NbCP,template.space,domain=None)
deform_op = odl.deform.LinDeformFixedTemplForward(template)

angle_partition = odl.uniform_partition(0, np.pi, 180)
detector_partition = odl.uniform_partition(-20, 20, 128)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)
ray_trafo = odl.tomo.RayTransform(templ_space, geometry)
deformed_template_true=deform_op(vector_field_op([CP,MOM]))
data = ray_trafo(deformed_template_true)
obj_fun = odl.solvers.L2NormSquared(ray_trafo.range) * (ray_trafo - data)


#%%


A=vector_field_op.derivative(X)
vect_field=vector_field_op(X)
A.adjoint(vect_field)

#%%

deform_app_CP=deform_op*vector_field_op
deform_app_CP(X).show()
deform_app_CP.derivative(X)(X).show()
vect_field_der_op=deform_app_CP.derivative(X)
X2=vect_field_der_op.adjoint(template)

#%%
CP=CPspace.element([[5,0],[10,10]])
MOM=MOMspace.element([[0,-10],[5,0]])
X=SpaceTot.element([CP,MOM])


function=obj_fun*deform_op*vector_field_op
function(X)
u=function.gradient(X)
#%%

scale_test=5

Xmax=round(40/(1*scale_test))
Ymax=round(40/(1*scale_test))
Xlin=numpy.linspace(-20,20,Xmax)
Ylin=numpy.linspace(-20,20,Ymax)
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

vector_field_op_test=odl.deform.CPVectorField(scale_test,NbCP_test,template.space,domain=None)


metric=odl.deform.RKHSNorm(scale_test,NbCP_test,SpaceTot_test,domain=None)
vector_field_op_test=odl.deform.CPVectorField(scale_test,NbCP_test,template.space,domain=None)

#functional=odl.deform.CPSmallDeformFixedTempl(vector_field_op_test,deform_op,10000*obj_fun,metric, domain=None)
functional=1000*obj_fun*deform_op*vector_field_op_test + metric
energy_init =functional([CP_test,MOM_test])
print(energy_init)
#%%
print('Initial energy {}'.format(energy_init))
line_search=odl.solvers.BacktrackingLineSearch(functional, tau=0.8, discount=0.00001, alpha=0.000010,
                 max_num_iter=20, estimate_step=True)
X0=SpaceTot_test.element([CP_test,MOM_test])
X=SpaceTot_test.element([CP_test,MOM_test])
A=odl.solvers.steepest_descent(functional, X, line_search, maxiter=50, tol=1e-16,
                     projection=None, callback=None)

energy_final=functional(X)
#%%
template.show('Initial template');
deformed_template_true.show('Deformed template ground truth');
deform_CP_op=deform_op*vector_field_op_test
deform_template_estimated=deform_CP_op(X)
deform_template_estimated.show('Estimated deformed template ');
deform_template_diff=deformed_template_true-template
deform_template_diff.show('Difference before optimisation');
deform_template_diff=deformed_template_true-deform_template_estimated
deform_template_diff.show('Difference after optimisation');
data_template = ray_trafo(template)
data_est = ray_trafo(deform_template_estimated)
data_template.show(title='Projection template (sinogram)');
data_est.show(title='Projection estimated deformed data (sinogram)');
data.show(title='Projection deformed data (ground truth, sinogram)');
residual_data=(data_template-data) #/(data+1)
residual_data.show('Initial differences differences');
residual_data=(data_est-data) #/(data+1)
residual_data.show('differences');

#%%




templ_space = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20], shape=[200, 200])
template = odl.phantom.shepp_logan(templ_space, modified=True)

NbCP=2
scale=10
CPspace=odl.ProductSpace(odl.rn(template.space.ndim), NbCP)
MOMspace=odl.ProductSpace(odl.rn(template.space.ndim), NbCP)
SpaceTot=odl.ProductSpace(CPspace,MOMspace)

vector_field_op=odl.deform.CPVectorField(scale,NbCP,template.space,domain=None)
deform_op = odl.deform.LinDeformFixedTemplForward(template)


deform_CP_op=odl.operator.OperatorComp(deform_op,vector_field_op)

deform_CP_op=deform_op*vector_field_op

CP=CPspace.element([[0,0],[10,10]])
MOM=MOMspace.element([[0,10],[0,0]])
deformed_template =deform_CP_op([CP,MOM])
deformed_template.show();
deformed_template_der =deform_CP_op.derivative([CP,MOM])
deformed_template_der([CP,MOM]).show();



angle_partition = odl.uniform_partition(0, np.pi, 180)
# Detector: uniformly sampled, n = 558, min = -30, max = 30
detector_partition = odl.uniform_partition(-20, 20, 128)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)



ray_trafo = odl.tomo.RayTransform(templ_space, geometry)
#ray_trafo = odl.tomo.RayTransform(templ_space, geometry, impl='astra_cuda')


# Create projection data by calling the ray transform on the deformed template
data = ray_trafo(deformed_template)
#data.show(title='Projection deformed data (sinogram)');

# Creating a function obj_fun which takes in input an image and returns the l2 difference between its ray transform and data
obj_fun = odl.solvers.L2NormSquared(ray_trafo.range) * (ray_trafo*deform_CP_op - data)

#%%

obj_fun.gradient([CP,MOM])


#%%













#%%

templ_space = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20], shape=[200, 200])
template = odl.phantom.shepp_logan(templ_space, modified=True)

NbCP=2
scale=10
CPspace=odl.ProductSpace(odl.rn(template.space.ndim), NbCP)
MOMspace=odl.ProductSpace(odl.rn(template.space.ndim), NbCP)

vector_field_op=odl.deform.CPSmallDeformFixedTempl(scale,NbCP, template, domain=None)

CP=CPspace.element([[0,0],[10,10]])
MOM=MOMspace.element([[0,10],[0,0]])
X=odl.ProductSpace(CPspace,MOMspace).element([CP,MOM])
NbCP=2
scale=1

deformed_template = vector_field_op(X)

deformed_template.show("deformed template");

#%%


vector_field_op=odl.deform.CPVectorField(scale,NbCP,template.space,domain=None)
vector_field=vector_field_op([CP,MOM])
vector_field_der=vector_field_op.derivative([CP,MOM])

vector_field_der([CP,MOM])

#%%

deform_CP_op=odl.operator.OperatorComp(deform_op,vector_field_op)

deformed_template =deform_CP_op([CP,MOM])
deformed_template.show()
deformed_template_der =deform_CP_op.derivative([CP,MOM])
deformed_template_der([CP,MOM]).show()

deformed_template_gradient =odl.discr.diff_ops.Gradient(deform_CP_op)([CP,MOM])
#%%
DerCP=CPspace.element([[0,0],[0,0]])
DerMOM=MOMspace.element([[0,0],[0,0]])
DerX=odl.ProductSpace(CPspace,MOMspace).element([DerCP,DerMOM])
vector_field_op.derivative(X)

#%%

n=2
NbCP=3
Space=odl.ProductSpace(odl.ProductSpace(odl.rn(n), NbCP), odl.ProductSpace(odl.rn(n), NbCP))

SpaceCP=odl.ProductSpace(odl.rn(n), NbCP)
SpaceMOM=odl.ProductSpace(odl.rn(n), NbCP)
Domain=odl.ProductSpace(SpaceCP,SpaceMOM)

CP=SpaceCP.element([[2,3],[2,3],[2,3]])
MOM=SpaceMOM.element([[2,3],[2,3],[2,3]])
X=Domain.element([CP,MOM])

from odl.discr import DiscreteLp, Gradient, Divergence
grad = Gradient(domain=template.space, method='central',pad_mode='symmetric')
grad_templ = grad(template)

#%%
sigma=scale

from odl.operator import Operator, PointwiseInner
domain = odl.ProductSpace(odl.ProductSpace(odl.rn(template.space.ndim), NbCP), odl.ProductSpace(odl.rn(template.space.ndim), NbCP))

def generate_disp_field_derivate(CP,MOM):
    # Returns an operator equal to the derivate of the generated vector field at [CP,MOM]
    # The intermediate computation tool derivate is list of two lists of vector fields
    # derivate[i][j][u] is the derivative w.r.t the u-th component of the j-th CP/MOM if i equals 0/1 (it is a list of 2 matrices, each one is the X/Y component of the displacement)

    disp_field_space = template.space.tangent_bundle
    derivateCP=odl.ProductSpace(odl.ProductSpace(disp_field_space, template.space.ndim),  NbCP).element()
    derivateMOM=odl.ProductSpace(odl.ProductSpace(disp_field_space, template.space.ndim),  NbCP).element()
    for k in range(0,NbCP):
        disp_func_grad = [
                          lambda x: ((x[0]-CP[k][0])/(sigma ** 2))*MOM[k][0]* np.exp(-((x[0]-CP[k][0]) ** 2 + (x[1]-CP[k][1])  ** 2) / (2 * sigma ** 2)),
                          lambda x: ((x[0]-CP[k][0])/(sigma ** 2))*MOM[k][1]* np.exp(-((x[0]-CP[k][0]) ** 2 + (x[1]-CP[k][1])  ** 2) / (2 * sigma ** 2))]

        disp_field_grad = disp_field_space.element(disp_func_grad)
        derivateCP[k][0]=disp_field_grad

        # Derivative of disp_field_est with respect to CP[k][1]
        disp_func_grad = [
                          lambda x: ((x[1]-CP[k][1])/(sigma ** 2))*MOM[k][0]* np.exp(-((x[0]-CP[k][0]) ** 2 + (x[1]-CP[k][1])  ** 2) / (2 * sigma ** 2)),
                          lambda x: ((x[1]-CP[k][1])/(sigma ** 2))*MOM[k][1]* np.exp(-((x[0]-CP[k][0]) ** 2 + (x[1]-CP[k][1])  ** 2) / (2 * sigma ** 2))]
        disp_field_grad = disp_field_space.element(disp_func_grad)
        derivateCP[k][1]=disp_field_grad

        # Derivative of disp_field_est with respect to MOM[k][0]
        disp_func_grad = [
                              lambda x: np.exp(-((x[0]-CP[k][0]) ** 2 + (x[1]-CP[k][1])  ** 2) / (2 * sigma ** 2)),
                              lambda x: 0]
        disp_field_grad = disp_field_space.element(disp_func_grad)
        derivateMOM[k][0]=disp_field_grad

        # Derivative of disp_field_est with respect to MOM[k][1]
        disp_func_grad = [
        lambda x: 0,
        lambda x: np.exp(-((x[0]-CP[k][0]) ** 2 + (x[1]-CP[k][1])  ** 2) / (2 * sigma ** 2))]
        disp_field_grad = disp_field_space.element(disp_func_grad)
        derivateMOM[k][1]=disp_field_grad

    domainDerivate=odl.ProductSpace(odl.ProductSpace(odl.ProductSpace(disp_field_space, template.space.ndim),  NbCP),2)
    derivate=domainDerivate.element([derivateCP,derivateMOM])
    # il faudrait retourner un operateur qui prend un Delta(CP,MOM)
    # et renvoie le champs de vecteurs egal à la combinaison lineaire
    # des champs de vect de derivate avec comme poids les composantes
    # de Delta(CP,MOM)
    return disp_field_space.element(PointwiseInner(domain, derivate))

#%%
domain=template.space.real_space.tangent_bundle
from odl.discr import DiscreteLp, Gradient, Divergence
def linear_deform(template, displacement, out=None):
    image_pts = template.space.points()
    for i, vi in enumerate(displacement):
        image_pts[:, i] += vi.ntuple.asarray()
    return template.interpolation(image_pts.T, out=out, bounds_check=False)

def derivativeVect(displacement):
    """Derivative of the operator at ``displacement``.

    Parameters
    ----------
    displacement : `domain` `element-like`
        Point at which the derivative is computed.

    Returns
    -------
    derivative : `PointwiseInner`
        The derivative evaluated at ``displacement``.
    """
    # To implement the complex case we need to be able to embed the real
    # vector field space into the range of the gradient. Issue #59.
#    displacement =domain.element(displacement)

    # TODO: allow users to select what method to use here.
    grad = Gradient(templ_space, method='central',
                    pad_mode='symmetric')
    grad_templ = grad(template)
    def_grad = domain.element(
        [linear_deform(gf, displacement) for gf in grad_templ])

    return PointwiseInner(domain, def_grad)
#%%
spc = odl.uniform_discr([-1, -1], [1, 1], (2, 3))
vfspace = odl.ProductSpace(spc, 2)
fixed_vf = np.array([[[0, 1,1],[0, 1,1]],[[1, -1,1],[0, 1,1]]])
pw_inner = odl.PointwiseInner(vfspace, fixed_vf)
x = vfspace.element([[[0,0,0]],[[0, 0,1]]])
#%%
spc = odl.uniform_discr([-1, -1], [1, 1], (2, 1))
vfspace = odl.ProductSpace(spc, 2)
fixed_vf = np.array([[[0],[1]],[[0],[0]]])
pw_inner = odl.PointwiseInner(vfspace, fixed_vf)
x = vfspace.element([[[0],[2]],[[0],[0]]])
pw_inner(x)