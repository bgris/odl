#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 10:22:34 2017

@author: bgris
"""


import odl
import numpy as np
from matplotlib import pylab as plt


#%%

#%%
########%% Generate template and ground truth for triangles

path='/home/bgris/Downloads/'

#I1name = '/home/bgris/Downloads/pictures/v.png'
#I0name = '/home/bgris/Downloads/pictures/j.png'
#I0 = np.rot90(plt.imread(I0name).astype('float'), -1)
#I1 = np.rot90(plt.imread(I1name).astype('float'), -1)
#
I0name = path + 'code_for_LDDMM---triangles/ss_save.png' # 438 * 438, I0[:,:,1]
I1name = path + 'code_for_LDDMM---triangles/ss_save_1.png' # 438 * 438, I0[:,:,1]
I0_init = np.rot90(plt.imread(I0name).astype('float'), -1)[:,:, 1]
I1_init = np.rot90(plt.imread(I1name).astype('float'), -1)[:,:, 1]


# Discrete reconstruction space: discretized functions on the rectangle
rec_space_init = odl.uniform_discr(
    min_pt=[-16, -16], max_pt=[16, 16], shape=[438, 438],
    dtype='float32', interp='linear')

rec_space = odl.uniform_discr(
    min_pt=[-16, -16], max_pt=[16, 16], shape=[256, 256],
    dtype='float32', interp='linear')

I0_init = rec_space_init.element(I0_init)
I1_init = rec_space_init.element(I1_init)

points = rec_space.points()

I0 = rec_space.element(I0_init.interpolation(points.T))
I1 = rec_space.element(I1_init.interpolation(points.T))

# Create the ground truth as the given image
ground_truth = rec_space.element(I0)*0.7 + 0.2

template =  rec_space.element(I1)*0.7 + 0.2


#%% Show

ground_truth.show(clim=[0,1])

#%% Save as array
path = '/home/bgris/data/Metamorphosis/'
name_exp = 'test1/'
name = path + name_exp
np.savetxt(name + 'template_values__0_2__0_9', template)
#%% Save as array
path = '/home/bgris/data/Metamorphosis/'
name_exp = 'test1/'
name = path + name_exp
np.savetxt(name + 'ground_truth_values__0_2__0_9', ground_truth)

#%% Load data

template_load = rec_space.element(np.loadtxt(name + '_values__0_2__0_9'))



#%% Generate data
name_list = ['ground_truth_values__0_2__0_9', 'ground_truth_values__0__1']
num_angles_list = [10, 50, 100]
maxiangle_list = ['pi', '0_25pi']
max_angle_list = [np.pi, 0.25*np.pi]
noise_level_list = [0.0, 0.05, 0.25]
noi_list = ['0', '0_05', '0_25']
min_angle = 0.0


for name_val in name_list:
    for num_angles in num_angles_list:
        for maxiangle, max_angle in zip(maxiangle_list, max_angle_list):
                for noi, noise_level in zip(noi_list, noise_level_list):
                        print(name_val)
                        print(num_angles)
                        print(maxiangle)
                        print(max_angle)
                        print(noi)
                        print(noise_level)
                        ## Create the uniformly distributed directions
                        angle_partition = odl.uniform_partition(min_angle, max_angle, num_angles,
                                                            nodes_on_bdry=[(True, True)])

                        ## Create 2-D projection domain
                        ## The length should be 1.5 times of that of the reconstruction space
                        detector_partition = odl.uniform_partition(-24, 24, int(round(rec_space.shape[0]*np.sqrt(2))))

                        ## Create 2-D parallel projection geometry
                        geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

                        ## Ray transform aka forward projection. We use ASTRA CUDA backend.
                        forward_op = odl.tomo.RayTransform(rec_space, geometry, impl='astra_cpu')

                        name_ground_truth = path + name_exp + name_val
                        ground_truth = rec_space.element(np.loadtxt(name_ground_truth))
                        name = name_ground_truth + 'num_angles_' + str(num_angles) + '_min_angle_0_max_angle_'
                        name += maxiangle + '_noise_' + noi

                        noise = noise_level * odl.phantom.noise.white_noise(forward_op.range)

                        data = forward_op(ground_truth) + noise

                        #data.show()

                        np.savetxt(name, data)


#
#%%

#%%
########%% Generate template and ground truth for SheppLogans


import odl
import numpy as np
from matplotlib import pylab as plt

from DeformationModulesODL.deform import Kernel
from DeformationModulesODL.deform import DeformationModuleAbstract
from DeformationModulesODL.deform import SumTranslations
from DeformationModulesODL.deform import UnconstrainedAffine
from DeformationModulesODL.deform import LocalScaling
from DeformationModulesODL.deform import LocalRotation
from DeformationModulesODL.deform import EllipseMvt
from DeformationModulesODL.deform import TemporalAttachmentModulesGeom
import scipy


rec_space = odl.uniform_discr(
    min_pt=[-16, -16], max_pt=[16, 16], shape=[256,256],
    dtype='float32', interp='linear')
## Values for SheppLogan0
val0=-0.7
val1=0.7
val2=0.5
ellipsoids=[[1.00, .6900, .9200, 0.0000, 0.0000, 0],
            [-.98, .6624, .8740, 0.0000, -.0184, 0],
            [val0, .1100, .3100, 0.2200, 0.0000, -18],
            [val0, .1600, .4100, -.2200, 0.0000, 18],
            [val1, .2100, .2500, 0.0000, 0.3500, 0],
            [val1, .0460, .0460, 0.0000, 0.1000, 0],
            [val1, .0460, .0460, 0.0000, -.1000, 0],
            [val2, .0460, .0230, -.0800, -.6050, 0],
            [val2, .0230, .0230, 0.0000, -.6060, 0],
            [val2, .0230, .0460, 0.0600, -.6050, 0]]

template=odl.phantom.ellipsoid_phantom(rec_space, ellipsoids)


## Values for SheppLogan1
background= 0.2
val0=-0.8 - background
val1=0.5 - background
val2=0.6 - background
ellipsoids=[[1.00, .6900, .9200, 0.0000, 0.0000, 0],
            [-.98, .6624, .8740, 0.0000, -.0184, 0],
            [val0, .1100, .3100, 0.2200, 0.0000, -18],
            [val0, .1600, .4100, -.2200, 0.0000, 18],
            [val1, .2100, .2500, 0.0000, 0.3500, 0],
            [val1, .0460, .0460, 0.0000, 0.1000, 0],
            [val1, .0460, .0460, 0.0000, -.1000, 0],
            [val2, .0460, .0230, -.0800, -.6050, 0],
            [val2, .0230, .0230, 0.0000, -.6060, 0],
            [val2, .0230, .0460, 0.0600, -.6050, 0]]

I0=odl.phantom.ellipsoid_phantom(rec_space, ellipsoids) + background

#template= odl.phantom.shepp_logan(space)
I0= rec_space.element(scipy.ndimage.filters.gaussian_filter(I0.asarray(),1))
#template.show(clim=[1,1.1])


NAffine=2
kernelaff=Kernel.GaussianKernel(3)
affine=UnconstrainedAffine.UnconstrainedAffine(rec_space, NAffine, kernelaff)

GD_affine=affine.GDspace.element([[-5,5],[3,4]])
Cont_affine=-1.2*affine.Contspace.element([[[0.5,0],[1,-1],[1,1]],[[-1,0.5],[-1,0],[0.5,0]]])
vect_field_affine=affine.ComputeField(GD_affine,Cont_affine)

I1=template.space.element(odl.deform.linearized._linear_deform(I0.copy(),vect_field_affine)).copy()
I2=template.space.element(odl.deform.linearized._linear_deform(template.copy(),vect_field_affine)).copy()
#I1_0.show(clim=[1,1.1])
mini=-1
maxi = 1
template.show('template', clim=[mini, maxi])
I0.show('I0', clim=[mini, maxi])
I1.show('I1', clim=[mini, maxi])
I2.show('I2', clim=[mini, maxi])

path = '/home/bgris/data/Metamorphosis/'
name_exp = 'test2/'
name = path + name_exp
np.savetxt(name + 'SheppLogan0', template)
np.savetxt(name + 'SheppLogan0_deformed', I2)
np.savetxt(name + 'SheppLogan1', I0)
np.savetxt(name + 'SheppLogan1_deformed', I1)


#%% Generate data
name_list = ['SheppLogan0_deformed', 'SheppLogan1_deformed']
num_angles_list = [10, 50, 100]
maxiangle_list = ['pi', '0_25pi']
max_angle_list = [np.pi, 0.25*np.pi]
noise_level_list = [0.0, 0.05, 0.25]
noi_list = ['0', '0_05', '0_25']
min_angle = 0.0


for name_val in name_list:
    for num_angles in num_angles_list:
        for maxiangle, max_angle in zip(maxiangle_list, max_angle_list):
                for noi, noise_level in zip(noi_list, noise_level_list):
                        print(name_val)
                        print(num_angles)
                        print(maxiangle)
                        print(max_angle)
                        print(noi)
                        print(noise_level)
                        ## Create the uniformly distributed directions
                        angle_partition = odl.uniform_partition(min_angle, max_angle, num_angles,
                                                            nodes_on_bdry=[(True, True)])

                        ## Create 2-D projection domain
                        ## The length should be 1.5 times of that of the reconstruction space
                        detector_partition = odl.uniform_partition(-24, 24, int(round(rec_space.shape[0]*np.sqrt(2))))

                        ## Create 2-D parallel projection geometry
                        geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

                        ## Ray transform aka forward projection. We use ASTRA CUDA backend.
                        forward_op = odl.tomo.RayTransform(rec_space, geometry, impl='astra_cpu')

                        name_ground_truth = path + name_exp + name_val
                        ground_truth = rec_space.element(np.loadtxt(name_ground_truth))
                        name = name_ground_truth + 'num_angles_' + str(num_angles) + '_min_angle_0_max_angle_'
                        name += maxiangle + '_noise_' + noi

                        noise = noise_level * odl.phantom.noise.white_noise(forward_op.range)

                        data = forward_op(ground_truth) + noise

                        #data.show()

                        np.savetxt(name, data)


#


#%%

#%%
########%% Generate template and ground truth for SheppLogans with different topology


import odl
import numpy as np
from matplotlib import pylab as plt

from DeformationModulesODL.deform import Kernel
from DeformationModulesODL.deform import DeformationModuleAbstract
from DeformationModulesODL.deform import SumTranslations
from DeformationModulesODL.deform import UnconstrainedAffine
from DeformationModulesODL.deform import LocalScaling
from DeformationModulesODL.deform import LocalRotation
from DeformationModulesODL.deform import EllipseMvt
from DeformationModulesODL.deform import TemporalAttachmentModulesGeom
import scipy


rec_space = odl.uniform_discr(
    min_pt=[-16, -16], max_pt=[16, 16], shape=[256,256],
    dtype='float32', interp='linear')
## Values for SheppLogan0
val0=-0.7
val1=0.7
val2=0.5
ellipsoids=[[1.00, .6900, .9200, 0.0000, 0.0000, 0],
            [-.98, .6624, .8740, 0.0000, -.0184, 0],
            [val0, .1600, .4100, -.2200, 0.0000, 18],
            [val1, .2100, .2500, 0.0000, 0.3500, 0],
            [val1, .0460, .0460, 0.0000, 0.1000, 0],
            [val1, .0460, .0460, 0.0000, -.1000, 0],
            [val2, .0460, .0230, -.0800, -.6050, 0],
            [val2, .0230, .0230, 0.0000, -.6060, 0],
            [val2, .0230, .0460, 0.0600, -.6050, 0]]

template=odl.phantom.ellipsoid_phantom(rec_space, ellipsoids)
template.show()
##%%

## Values for SheppLogan1
background= 0.2
val0=-0.8 - background
val1=0.5 - background
val2=0.6 - background
ellipsoids=[[1.00, .6900, .9200, 0.0000, 0.0000, 0],
            [-.98, .6624, .8740, 0.0000, -.0184, 0],
            [val0, .1600, .4100, -.2200, 0.0000, 18],
            [val1, .2100, .2500, 0.0000, 0.3500, 0],
            [val1, .0460, .0460, 0.0000, 0.1000, 0],
            [val1, .0460, .0460, 0.0000, -.1000, 0],
            [val2, .0460, .0230, -.0800, -.6050, 0],
            [val2, .0230, .0230, 0.0000, -.6060, 0],
            [val2, .0230, .0460, 0.0600, -.6050, 0]]

I0=odl.phantom.ellipsoid_phantom(rec_space, ellipsoids) + background

#template= odl.phantom.shepp_logan(space)
I0= rec_space.element(scipy.ndimage.filters.gaussian_filter(I0.asarray(),1))
#template.show(clim=[1,1.1])


NAffine=2
kernelaff=Kernel.GaussianKernel(3)
affine=UnconstrainedAffine.UnconstrainedAffine(rec_space, NAffine, kernelaff)

GD_affine=affine.GDspace.element([[-5,5],[3,4]])
Cont_affine=-1.2*affine.Contspace.element([[[0.5,0],[1,-1],[1,1]],[[-1,0.5],[-1,0],[0.5,0]]])
vect_field_affine=affine.ComputeField(GD_affine,Cont_affine)

I1=template.space.element(odl.deform.linearized._linear_deform(I0.copy(),vect_field_affine)).copy()
I2=template.space.element(odl.deform.linearized._linear_deform(template.copy(),vect_field_affine)).copy()
#I1_0.show(clim=[1,1.1])
mini=-1
maxi = 1
template.show('template', clim=[mini, maxi])
I0.show('I0', clim=[mini, maxi])
I1.show('I1', clim=[mini, maxi])
I2.show('I2', clim=[mini, maxi])

path = '/home/bgris/data/Metamorphosis/'
name_exp = 'test3/'
name = path + name_exp
np.savetxt(name + 'SheppLogan2', template)
np.savetxt(name + 'SheppLogan2_deformed', I2)
np.savetxt(name + 'SheppLogan3', I0)
np.savetxt(name + 'SheppLogan3_deformed', I1)


#%% Generate data
name_list = ['SheppLogan2_deformed', 'SheppLogan3_deformed']
num_angles_list = [10, 50, 100]
maxiangle_list = ['pi', '0_25pi']
max_angle_list = [np.pi, 0.25*np.pi]
noise_level_list = [0.0, 0.05, 0.25]
noi_list = ['0', '0_05', '0_25']
min_angle = 0.0


for name_val in name_list:
    for num_angles in num_angles_list:
        for maxiangle, max_angle in zip(maxiangle_list, max_angle_list):
                for noi, noise_level in zip(noi_list, noise_level_list):
                        print(name_val)
                        print(num_angles)
                        print(maxiangle)
                        print(max_angle)
                        print(noi)
                        print(noise_level)
                        ## Create the uniformly distributed directions
                        angle_partition = odl.uniform_partition(min_angle, max_angle, num_angles,
                                                            nodes_on_bdry=[(True, True)])

                        ## Create 2-D projection domain
                        ## The length should be 1.5 times of that of the reconstruction space
                        detector_partition = odl.uniform_partition(-24, 24, int(round(rec_space.shape[0]*np.sqrt(2))))

                        ## Create 2-D parallel projection geometry
                        geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

                        ## Ray transform aka forward projection. We use ASTRA CUDA backend.
                        forward_op = odl.tomo.RayTransform(rec_space, geometry, impl='astra_cpu')

                        name_ground_truth = path + name_exp + name_val
                        ground_truth = rec_space.element(np.loadtxt(name_ground_truth))
                        name = name_ground_truth + 'num_angles_' + str(num_angles) + '_min_angle_0_max_angle_'
                        name += maxiangle + '_noise_' + noi

                        noise = noise_level * odl.phantom.noise.white_noise(forward_op.range)

                        data = forward_op(ground_truth) + noise

                        #data.show()

                        np.savetxt(name, data)


#




