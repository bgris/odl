#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 11:46:41 2017

@author: bgris
"""

""" test for simulated 4D cardiac data """

import odl
import numpy as np
from matplotlib import pylab as plt


#%% Load data as a list of images
a=120
b=256
c=256

space = odl.uniform_discr(
    min_pt=[-127, -127, -59], max_pt=[127, 127, 59], shape=[a,b,c],
    dtype='float32', interp='linear')

Ndata=2
data_list=[]
index_list=[0,4]
for i in range(len(index_list)+1):
    filename='SPECT_Torso_act_'+ str(index_list[i]+1) + '.bin'
    A = np.fromfile(filename, dtype='float32')
    A = A.reshape([a,b,c])
    data_list.append(space.element(A))

data_list[0].show(indices=np.s_[ space.shape[0] // 2,:, :])

data_list[1].show(indices=np.s_[ space.shape[0] // 2,:, :])

#%% Parameter for matching
forward_op=odl.IdentityOperator(space)
nb_time_point_int=2
template=data_list[0]
data_time_points=np.array([1])
data_space=odl.ProductSpace(forward_op.range,data_time_points.size)
data=data_space.element([forward_op(data_list[1])])
forward_operators=[forward_op]
Norm=odl.solvers.L2NormSquared(forward_op.range)
def kernel(x):
    sigma = 1.0
    scaled = [xi ** 2 / (2 * sigma ** 2) for xi in x]
    return np.exp(-sum(scaled))



energy_op=odl.deform.TemporalAttachmentLDDMMGeom(nb_time_point_int, template ,data,
                            data_time_points, forward_operators,Norm, kernel,
                            domain=None)


Reg=odl.deform.RegularityLDDMM(kernel,energy_op.domain)

lam= 1e-8

functional = energy_op + lam*Reg

#%% Initialization

vector_fields_list_init=energy_op.domain.zero()

vector_fields_list=vector_fields_list_init.copy()

niter=60
eps = 0.000001




#%% launch matching
attachment_term=energy_op(vector_fields_list)
print(" Initial ,  attachment term : {}".format(attachment_term))

mod=1

for k in range(niter):
    if (mod==1):
        grad=functional.gradient(vector_fields_list)

    vector_fields_list_temp= (vector_fields_list- eps *grad).copy()
    attachment_term_temp=energy_op(vector_fields_list_temp)

    if attachment_term_temp< attachment_term:
        vector_fields_list=vector_fields_list_temp.copy()
        attachment_term = attachment_term_temp
        print(" iter : {}  ,  attachment_term : {}".format(k,attachment_term))
        mod=1
        eps = eps*1.2
    else:
        eps = eps*0.8
        print(" iter : {}  ,  epsilon : {}, attachment_term_temp : {}".format(k,eps,attachment_term_temp))
        mod=0
#


#%% See result

Registration=odl.deform.ShootTemplateFromVectorFields(vector_fields_list, template)

template.show('Initial template',indices=np.s_[ space.shape[0] // 2,:, :])
data[data_time_points.size-1].show('Target',indices=np.s_[ space.shape[0] // 2,:, :])
Registration[nb_time_point_int].show('Result, final time',indices=np.s_[ space.shape[0] // 2,:, :])
((data[data_time_points.size-1]-template)**2).show('Initial Difference',indices=np.s_[ space.shape[0] // 2,:, :])
((data[data_time_points.size-1]-Registration[nb_time_point_int])**2).show('Difference, final time',indices=np.s_[ space.shape[0] // 2,:, :])


