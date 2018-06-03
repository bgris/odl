#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 14:15:11 2018

@author: bgris
"""


import odl
import numpy as np
from matplotlib import pylab as plt
import os

space = odl.uniform_discr(
    min_pt=[-16, -16], max_pt=[16, 16], shape=[512, 512],
    dtype='float32', interp='linear')


points = space.points().T
template = 1 * space.element(points[0]< 10)
ground_truth = 1 * space.element(points[0]< -10)

sigma = 0.5

# Give kernel function
def kernel(x):
    scaled = [xi ** 2 / (2 * sigma ** 2) for xi in x]
    return np.exp(-sum(scaled))


forward_op = odl.IdentityOperator(space)

## load data

data= forward_op(ground_truth)
lamb = 1e5
tau = 1e-5


data=[data]
#data=[proj_data]
data_time_points=np.array([1])
forward_operators=[forward_op]
Norm=odl.solvers.L2NormSquared(forward_op.range)

nb_time_point_int = 20
##%% Define energy operator

functional=odl.deform.TemporalAttachmentMetamorphosisGeom(nb_time_point_int,
                            lamb,tau,template ,data,
                            data_time_points, forward_operators,Norm, kernel,
                            domain=None)


##%%
##%% Gradient descent

X_init=functional.domain.zero()
X=X_init.copy()
energy=functional(X)
#%% Gradient descent
niter = 300
epsV=0.02
epsZ=0.02
print(" Initial ,  energy : {}".format(energy))


for k in range(niter):
    #X[1][0].show("Iter = {}".format(k),clim=[-1,1])
    #grad=functional.gradient(X)
    grad=functional.gradient(X)
    X_temp0=X.copy()
    X_temp0[0]= (X[0]- epsV *grad[0]).copy()
    X_temp0[1]= (X[1]- epsZ *grad[1]).copy()
    energy_temp0=functional(X_temp0)
    if energy_temp0<energy:
        X=X_temp0.copy()
        energy=energy_temp0
        epsV*=1.1
        epsZ*=1.1
        print(" iter : {}  , energy : {}, epsV = {} , epsZ = {}".format(k,energy,epsV, epsZ))
    else:
        X_temp1=X.copy()
        X_temp1[0]= (X[0]- epsV *grad[0]).copy()
        X_temp1[1]= (X[1]- 0.5*epsZ *grad[1]).copy()
        energy_temp1=functional(X_temp1)

        X_temp2=X.copy()
        X_temp2[0]= (X[0]- 0.5*epsV *grad[0]).copy()
        X_temp2[1]= (X[1]- epsZ *grad[1]).copy()
        energy_temp2=functional(X_temp2)

        X_temp3=X.copy()
        X_temp3[0]= (X[0]- 0.5*epsV *grad[0]).copy()
        X_temp3[1]= (X[1]- 0.5*epsZ *grad[1]).copy()
        energy_temp3=functional(X_temp3)

        if (energy_temp3<=energy_temp1 and energy_temp3<=energy_temp2):
            X_temp0=X_temp3.copy()
            energy_temp0=energy_temp3
            epsZ*=0.5
            epsV*=0.5
        else:
            if (energy_temp1<=energy_temp2 and energy_temp1<=energy_temp3):
                X_temp0=X_temp1.copy()
                energy_temp0=energy_temp1
                epsZ*=0.5
            else:
                X_temp0=X_temp2.copy()
                energy_temp0=energy_temp2
                epsV*=0.5

        if energy_temp0<energy:
            X=X_temp0.copy()
            energy=energy_temp0
            epsV *= 1.1
            epsZ *= 1.1
            print(" iter : {}  , energy : {}, epsV = {} , epsZ = {}".format(k,energy,epsV, epsZ))
        else:
            print("epsV = {} , epsZ = {}".format(epsV, epsZ))

#    if (np.floor_divide(k,5) == 0):
#        plt.close('all')
#        functional.ComputeMetamorphosis(X[0], X[1])[-1].show(str(k))

#%%



##%% Compute estimated trajectory
image_list_data=functional.ComputeMetamorphosis(X[0],X[1])


image_list=functional.ComputeMetamorphosisListInt(X[0],X[1])

template_evo=odl.deform.ShootTemplateFromVectorFields(X[0], template)

zeta_transp=odl.deform.ShootSourceTermBackwardlist(X[0],X[1])

image_evol=odl.deform.IntegrateTemplateEvol(template,zeta_transp,0,nb_time_point_int)
##%%
mini=-1
maxi=1
#


for i in range(nb_time_point_int):
    image_list[i].show('metamorphosis ' + str(i))


for i in range(nb_time_point_int):
    image_evol[i].show('image ' + str(i))


for i in range(nb_time_point_int):
    template_evo[i].show('template ' + str(i))


#    image_list[i].show('metamorphosis ' + str(i))
#    image_list[i].show('metamorphosis ' + str(i))
