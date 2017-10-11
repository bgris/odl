#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 18:09:53 2017

@author: bgris
"""

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

import nibabel as nib
import os
import scipy
#%% Load data as a list of images
a=120
b=256
c=256

space = odl.uniform_discr(
    min_pt=[-127, -127, -59], max_pt=[127, 127, 59], shape=[a,b,c],
    dtype='float32', interp='linear')


data_list=[]
index_list=[0,4]
for i in range(len(index_list)):
    filename='/home/bgris/odl/examples/CardiacPhantom/SPECT_Torso_act_'+ str(index_list[i]+1) + '.bin'
    A = np.fromfile(filename, dtype='float32')
    A = A.reshape([a,b,c]).copy()
    data_list.append(space.element(A))
Ndata=len(index_list)
data_list[0].show(indices=np.s_[ space.shape[0] // 2,:, :])

data_list[1].show(indices=np.s_[ space.shape[0] // 2,:, :])

# Save data to visualize them
if False:
    indi=7
    filename='/home/bgris/odl/examples/CardiacPhantom/SPECT_Torso_act_'+ str(indi+1) + '.bin'
    A = np.fromfile(filename, dtype='float32')
    A = A.reshape([a,b,c]).copy()
    #A=data_list[1]
    data = np.asarray(A)
    img = nib.Nifti1Image(data, np.eye(4))
    img.get_data_dtype() == np.dtype(np.float32)
    img.header.get_xyzt_units()
    img.to_filename(os.path.join('/home/bgris/odl/examples/CardiacPhantom','Data_noise'+ str(indi+1) + '.nii'))
#


#%% Parameter for matching
forward_op=odl.IdentityOperator(space)
nb_time_point_int=10
template=data_list[0]
template=template.space.element(scipy.ndimage.filters.gaussian_filter(data_list[0].asarray(),1))
data_time_points=np.array([1])
data_space=odl.ProductSpace(forward_op.range,data_time_points.size)
#data=data_space.element([forward_op(data_list[1])])
data=data_space.element([forward_op(space.element(scipy.ndimage.filters.gaussian_filter(data_list[i].asarray(),1))) for i in range(1,len(index_list))])
forward_operators=[forward_op,forward_op]
Norm=odl.solvers.L2NormSquared(forward_op.range)
def kernel(x):
    sigma = 10.0
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

niter=300
eps = 1e-6




#%% launch matching
attachment_term=energy_op(vector_fields_list)
print(" Initial ,  attachment term : {}".format(attachment_term))

mod=1
grad_op=functional.gradient

for k in range(niter):
    if (mod==1):
        grad=grad_op(vector_fields_list)

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
index_time_data=[10]
for i in range(len(index_list)):
    data[i].show('Ground truth time{}'.format(i),indices=np.s_[ space.shape[0] // 2,:, :])
    Registration[index_time_data[i]].show('Result time{}'.format(i),indices=np.s_[ space.shape[0] // 2,:, :])
    ((data[i]-template)**2).show('Initial Difference time{}'.format(i),indices=np.s_[ space.shape[0] // 2,:, :])
    ((data[i]-Registration[index_time_data[i]])**2).show('Final ifference,  time{}'.format(i),indices=np.s_[ space.shape[0] // 2,:, :])

#%%
for i in range(nb_time_point_int):
    Registration[i].show('Result time{}'.format(i),indices=np.s_[ space.shape[0] // 2,:, :])

#%% save registration to see
for i in range(nb_time_point_int):
    A=Registration[i].copy()
    #A=data[0]
    A = np.asarray(A)
    img = nib.Nifti1Image(A, np.eye(4))
    img.get_data_dtype() == np.dtype(np.float32)
    img.header.get_xyzt_units()
    img.to_filename(os.path.join('/home/bgris/odl/examples/CardiacPhantom','Registration_matching_1_5_'+ str(i+1) + '.nii'))


#%% save results

for t in range(nb_time_point_int+1):
    np.savetxt('/home/bgris/odl/examples/CardiacPhantom/LDDMM_matching_1_5__Vector_Field_time_{}'.format(t),vector_fields_list[t])
