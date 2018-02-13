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
#num_angles_list = [10, 50, 100]
num_angles_list = [6]
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


#%%
########%% Generate template and ground truth for SheppLogans with
# different and non uniform intensities and 'tumor'


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


ellipsoids=[[1.00, .6900, .9200, 0.0000, 0.0000, 0]]

mask=odl.phantom.ellipsoid_phantom(rec_space, ellipsoids)

mask.show()

noise =0.5 * odl.phantom.noise.white_noise(rec_space)
mask_noise = (noise*(1-mask))
mask_noise.show()



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
template.show()


mask2 = (np.array(template )== 0) + ((np.array(template) < 0.03)*(np.array(template) > 0.01))
mask2 = rec_space.element(mask2)
mask2.show()
noise =0.5 * odl.phantom.noise.white_noise(rec_space)
mask_noise2 = (noise*(mask2))
mask_noise2.show()
##%%

## Values for SheppLogan1
background= 0.1
val0=-0.8 - background
val1=0.5 - background
val2=0.6 - background
val3 = 1-background

ellipsoids=[[1.00, .6900, .9200, 0.0000, 0.0000, 0],
            [-.98, .6624, .8740, 0.0000, -.0184, 0],
            [val0, .1100, .3100, 0.2200, 0.0000, -18],
            [val0, .1600, .4100, -.2200, 0.0000, 18],
            [val1, .2100, .2500, 0.0000, 0.3500, 0],
            [val1, .0460, .0460, 0.0000, 0.1000, 0],
            [val1, .0460, .0460, 0.0000, -.1000, 0],
            [val2, .0460, .0230, -.0800, -.6050, 0],
            [val2, .0230, .0230, 0.0000, -.6060, 0],
            [val2, .0230, .0460, 0.0600, -.6050, 0],
            [val3, .0460, .0460, 0.400, .4050, 0]]

I0=odl.phantom.ellipsoid_phantom(rec_space, ellipsoids) + background + mask_noise

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
#%%
path = '/home/bgris/data/Metamorphosis/'
name_exp = 'test6/'
name = path + name_exp
np.savetxt(name + 'SheppLogan4', I0)
np.savetxt(name + 'SheppLogan4_deformed', I1)


#%% Generate data
name_list = ['SheppLogan4_deformed']
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
########%% Generate template and several ground truths (for time dependent problems)
# for SheppLogans with
# different and non uniform intensities and 'tumor'


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


ellipsoids=[[1.00, .6900, .9200, 0.0000, 0.0000, 0]]

mask=odl.phantom.ellipsoid_phantom(rec_space, ellipsoids)

mask.show()

noise =0.5 * odl.phantom.noise.white_noise(rec_space)
mask_noise = (noise*(1-mask))
mask_noise.show()


## Values for SheppLogan1
background= 0.1
val0=-0.8 - background
val1=0.5 - background
val2=0.6 - background
val3 = 1-background

ellipsoidsinit=[[1.00, .6900, .9200, 0.0000, 0.0000, 0],
            [-.98, .6624, .8740, 0.0000, -.0184, 0],
            [val0, .1100, .3100, 0.2200, 0.0000, -18],
            [val0, .1600, .4100, -.2200, 0.0000, 18],
            [val1, .2100, .2500, 0.0000, 0.3500, 0],
            [val1, .0460, .0460, 0.0000, 0.1000, 0],
            [val1, .0460, .0460, 0.0000, -.1000, 0],
            [val2, .0460, .0230, -.0800, -.6050, 0],
            [val2, .0230, .0230, 0.0000, -.6060, 0],
            [val2, .0230, .0460, 0.0600, -.6050, 0]]

template=odl.phantom.ellipsoid_phantom(rec_space, ellipsoidsinit) + background + mask_noise

#template= odl.phantom.shepp_logan(space)
template= rec_space.element(scipy.ndimage.filters.gaussian_filter(template.asarray(),1))

## Values for SheppLogan1
background= 0.1
val0=-0.8 - background
val1=0.5 - background
val2=0.6 - background
val3 = 1-background

ellipsoids0=[[1.00, .6900, .9200, 0.0000, 0.0000, 0],
            [-.98, .6624, .8740, 0.0000, -.0184, 0],
            [val0, .1100, .3100, 0.2200, 0.0000, -18],
            [val0, .1600, .4100, -.2200, 0.0000, 18],
            [val1, .2100, .2500, 0.0000, 0.3500, 0],
            [val1, .0460, .0460, 0.0000, 0.1000, 0],
            [val1, .0460, .0460, 0.0000, -.1000, 0],
            [val2, .0460, .0230, -.0800, -.6050, 0],
            [val2, .0230, .0230, 0.0000, -.6060, 0],
            [val2, .0230, .0460, 0.0600, -.6050, 0],
            [val3, .0460, .0460, 0.400, .4050, 0]]

I0=odl.phantom.ellipsoid_phantom(rec_space, ellipsoids0) + background + mask_noise

#template= odl.phantom.shepp_logan(space)
I0= rec_space.element(scipy.ndimage.filters.gaussian_filter(I0.asarray(),1))
#template.show(clim=[1,1.1])

ellipsoids1=[[1.00, .6900, .9200, 0.0000, 0.0000, 0],
            [-.98, .6624, .8740, 0.0000, -.0184, 0],
            [val0, .1100, .3100, 0.2200, 0.0000, -18],
            [val0, .1600, .4100, -.2200, 0.0000, 18],
            [val1, .2100, .2500, 0.0000, 0.3500, 0],
            [val1, .0460, .0460, 0.0000, 0.1000, 0],
            [val1, .0460, .0460, 0.0000, -.1000, 0],
            [val2, .0460, .0230, -.0800, -.6050, 0],
            [val2, .0230, .0230, 0.0000, -.6060, 0],
            [val2, .0230, .0460, 0.0600, -.6050, 0],
            [val3, .0260, .0260, 0.400, .4050, 0]]

I0bis=odl.phantom.ellipsoid_phantom(rec_space, ellipsoids1) + background + mask_noise

#template= odl.phantom.shepp_logan(space)
I0bis= rec_space.element(scipy.ndimage.filters.gaussian_filter(I0bis.asarray(),1))
#template.show(clim=[1,1.1])


NAffine=2
kernelaff=Kernel.GaussianKernel(3)
affine=UnconstrainedAffine.UnconstrainedAffine(rec_space, NAffine, kernelaff)

GD_affine=affine.GDspace.element([[-5,5],[3,4]])
Cont_affine=-1.2*affine.Contspace.element([[[0.5,0],[1,-1],[1,1]],[[-1,0.5],[-1,0],[0.5,0]]])
vect_field_affine=affine.ComputeField(GD_affine,Cont_affine)

I1=rec_space.element(odl.deform.linearized._linear_deform(I0bis.copy(),0.75*vect_field_affine)).copy()
I2=rec_space.element(odl.deform.linearized._linear_deform(I0.copy(),1.2*vect_field_affine)).copy()
#I2=template.space.element(odl.deform.linearized._linear_deform(template.copy(),vect_field_affine)).copy()
#I1_0.show(clim=[1,1.1])
mini=-1
maxi = 1
template.show('template', clim=[mini, maxi])
I0.show('I0', clim=[mini, maxi])
I1.show('I1', clim=[mini, maxi])
I2.show('I2', clim=[mini, maxi])
#%%
path = '/home/bgris/data/Metamorphosistemporal/'
name_exp = 'test7/'
name = path + name_exp
np.savetxt(name + 'SheppLogan7_0', template)
np.savetxt(name + 'SheppLogan7_1', I1)
np.savetxt(name + 'SheppLogan7_2', I2)


#%% Generate data
name_list = ['SheppLogan7_0', 'SheppLogan7_1', 'SheppLogan7_2']
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
########%% Generate template and ground truth for SheppLogans with
# different and non uniform intensities without 'tumor'


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


ellipsoids=[[1.00, .6900, .9200, 0.0000, 0.0000, 0]]

mask=odl.phantom.ellipsoid_phantom(rec_space, ellipsoids)

mask.show()

noise =0.5 * odl.phantom.noise.white_noise(rec_space)
mask_noise = (noise*(1-mask))
mask_noise.show()



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
template.show()


mask2 = (np.array(template )== 0) + ((np.array(template) < 0.03)*(np.array(template) > 0.01))
mask2 = rec_space.element(mask2)
mask2.show()
noise =0.5 * odl.phantom.noise.white_noise(rec_space)
mask_noise2 = (noise*(mask2))
mask_noise2.show()
##%%

## Values for SheppLogan1
background= 0.1
val0=-0.8 - background
val1=0.5 - background
val2=0.6 - background
val3 = 1-background

ellipsoids=[[1.00, .6900, .9200, 0.0000, 0.0000, 0],
            [-.98, .6624, .8740, 0.0000, -.0184, 0],
            [val0, .1100, .3100, 0.2200, 0.0000, -18],
            [val0, .1600, .4100, -.2200, 0.0000, 18],
            [val1, .2100, .2500, 0.0000, 0.3500, 0],
            [val1, .0460, .0460, 0.0000, 0.1000, 0],
            [val1, .0460, .0460, 0.0000, -.1000, 0],
            [val2, .0460, .0230, -.0800, -.6050, 0],
            [val2, .0230, .0230, 0.0000, -.6060, 0],
            [val2, .0230, .0460, 0.0600, -.6050, 0],]

I0=odl.phantom.ellipsoid_phantom(rec_space, ellipsoids) + background + mask_noise

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
#%%
path = '/home/bgris/data/Metamorphosis/'
name_exp = 'test7/'
name = path + name_exp
np.savetxt(name + 'SheppLogan5', I0)
np.savetxt(name + 'SheppLogan5_deformed', I1)


#%% Generate data
name_list = ['SheppLogan5_deformed']
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
########%% Generate template and ground truth for SheppLogans with
# different and non uniform intensities and 'tumor' on the left (value of the tumor to choose)


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


ellipsoids=[[1.00, .6900, .9200, 0.0000, 0.0000, 0]]

mask=odl.phantom.ellipsoid_phantom(rec_space, ellipsoids)

mask.show()

noise =0.5 * odl.phantom.noise.white_noise(rec_space)
mask_noise = (noise*(1-mask))
mask_noise.show()



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
template.show()


mask2 = (np.array(template )== 0) + ((np.array(template) < 0.03)*(np.array(template) > 0.01))
mask2 = rec_space.element(mask2)
mask2.show()
noise =0.5 * odl.phantom.noise.white_noise(rec_space)
mask_noise2 = (noise*(mask2))
mask_noise2.show()
##%%

## Values for SheppLogan1
background= 0.1
val0=-0.8 - background
val1=0.5 - background
val2=0.6 - background
val3 = 0.6-background

ellipsoids=[[1.00, .6900, .9200, 0.0000, 0.0000, 0],
            [-.98, .6624, .8740, 0.0000, -.0184, 0],
            [val0, .1100, .3100, 0.2200, 0.0000, -18],
            [val0, .1600, .4100, -.2200, 0.0000, 18],
            [val1, .2100, .2500, 0.0000, 0.3500, 0],
            [val1, .0460, .0460, 0.0000, 0.1000, 0],
            [val1, .0460, .0460, 0.0000, -.1000, 0],
            [val2, .0460, .0230, -.0800, -.6050, 0],
            [val2, .0230, .0230, 0.0000, -.6060, 0],
            [val2, .0230, .0460, 0.0600, -.6050, 0],
            [val3, .0460, .0460, -0.540, .3050, 0]]

I0=odl.phantom.ellipsoid_phantom(rec_space, ellipsoids) + background + mask_noise

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
#%%
path = '/home/bgris/data/Metamorphosis/'
name_exp = 'test10/'
name = path + name_exp
np.savetxt(name + 'SheppLogan9', I0)
np.savetxt(name + 'SheppLogan9_deformed', I1)


#%% Generate data
name_list = ['SheppLogan9_deformed']
num_angles_list = [10, 50, 100, 20]
#num_angles_list = [20, 30]
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
########%% Generate template and ground truth for SheppLogans with
# different and non uniform intensities and 'tumor' on the left (value of the tumor to choose)
# real 'sheppLogan' values !


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


ellipsoids=[[1.00, .6900, .9200, 0.0000, 0.0000, 0]]

mask=odl.phantom.ellipsoid_phantom(rec_space, ellipsoids)

mask.show()

noise =0.5 * odl.phantom.noise.white_noise(rec_space)
mask_noise = (noise*(1-mask))
mask_noise.show()



## Values for SheppLogan0
val0=-0.2
val1=0.1
val2=0.1

ellipsoids=[[1.00, .6900, .9200, 0.0000, 0.0000, 0],
            [-.8, .6624, .8740, 0.0000, -.0184, 0],
            [val0, .1100, .3100, 0.2200, 0.0000, -18],
            [val0, .1600, .4100, -.2200, 0.0000, 18],
            [val1, .2100, .2500, 0.0000, 0.3500, 0],
            [val1, .0460, .0460, 0.0000, 0.1000, 0],
            [val1, .0460, .0460, 0.0000, -.1000, 0],
            [val2, .0460, .0230, -.0800, -.6050, 0],
            [val2, .0230, .0230, 0.0000, -.6060, 0],
            [val2, .0230, .0460, 0.0600, -.6050, 0]]

template=odl.phantom.ellipsoid_phantom(rec_space, ellipsoids)
#template = odl.phantom.shepp_logan(rec_space, modified=True)
template.show()

#%%
mask2 = (np.array(template )== 0) + ((np.array(template) < 0.03)*(np.array(template) > 0.01))
mask2 = rec_space.element(mask2)
mask2.show()
noise =0.5 * odl.phantom.noise.white_noise(rec_space)
mask_noise2 = (noise*(mask2))
mask_noise2.show()
##%%

## Values for SheppLogan1
background= 0.0
val0=-0.2 - background
val1=0.1 - background
val2=0.1 - background
val3 = 0.3-background

#[1.0, -0.8, -0.2, -0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
nb_data_points = 10

NAffine=2
kernelaff=Kernel.GaussianKernel(3)
affine=UnconstrainedAffine.UnconstrainedAffine(rec_space, NAffine, kernelaff)

GD_affine=affine.GDspace.element([[-5,5],[3,4]])
Cont_affine=-1.5*affine.Contspace.element([[[0.5,0],[1,-1],[1,1]],[[-1.5, 1],[-2,0],[1,0]]])
vect_field_affine=affine.ComputeField(GD_affine,Cont_affine)
data_list = []

ellipsoids=[[1.00, .6900, .9200, 0.0000, 0.0000, 0],
                [-.8, .6624, .8740, 0.0000, -.0184, 0],
                [val0, .1100, .3100, 0.2200, 0.0000, -18],
                [val0, .1600, .4100, -.2200, 0.0000, 18],
                [val1, .2100, .2500, 0.0000, 0.3500, 0],
                [val1, .0460, .0460, 0.0000, 0.1000, 0],
                [val1, .0460, .0460, 0.0000, -.1000, 0],
                [val2, .0460, .0230, -.0800, -.6050, 0],
                [val2, .0230, .0230, 0.0000, -.6060, 0],
                [val2, .0230, .0460, 0.0600, -.6050, 0]]
I0=odl.phantom.ellipsoid_phantom(rec_space, ellipsoids) + background + mask_noise

data_list.append(I0.copy())
for i in range(nb_data_points):
    fac = (i+1)/nb_data_points
    ellipsoids=[[1.00, .6900, .9200, 0.0000, 0.0000, 0],
                [-.8, .6624, .8740, 0.0000, -.0184, 0],
                [val0, .1100, .3100, 0.2200, 0.0000, -18],
                [val0, .1600, .4100, -.2200, 0.0000, 18],
                [val1, .2100, .2500, 0.0000, 0.3500, 0],
                [val1, .0460, .0460, 0.0000, 0.1000, 0],
                [val1, .0460, .0460, 0.0000, -.1000, 0],
                [val2, .0460, .0230, -.0800, -.6050, 0],
                [val2, .0230, .0230, 0.0000, -.6060, 0],
                [val2, .0230, .0460, 0.0600, -.6050, 0],
                [fac*val3, fac*.046, fac*.046, -0.540, .3050, 0]]

    I0=odl.phantom.ellipsoid_phantom(rec_space, ellipsoids) + background + mask_noise

    #template= odl.phantom.shepp_logan(space)
    I0= rec_space.element(scipy.ndimage.filters.gaussian_filter(I0.asarray(),1))
    #template.show(clim=[1,1.1])


    I1=template.space.element(odl.deform.linearized._linear_deform(I0.copy(),fac*vect_field_affine)).copy()
    data_list.append(I1.copy())
    #I2=template.space.element(odl.deform.linearized._linear_deform(template.copy(),fac*vect_field_affine)).copy()
#I1_0.show(clim=[1,1.1])
mini=0
maxi = 1

for i in range(nb_data_points+1):
    data_list[i].show(str(i), clim=[mini, maxi])
    
#template.show('template', clim=[mini, maxi])
#I0.show('I0', clim=[mini, maxi])
#I1.show('I1', clim=[mini, maxi])
#I2.show('I2', clim=[mini, maxi])
#%%

path = '/home/barbara/data/Metamorphosis/'
name_exp = 'test13/'
name = path + name_exp
for i in range(nb_data_points+1):
    np.savetxt(name + 'temporal__t_' + str(i), data_list[i])
#np.savetxt(name + 'SheppLogan10', template)
#np.savetxt(name + 'SheppLogan10_deformed', I2)
#np.savetxt(name + 'SheppLogan13', I0)
#np.savetxt(name + 'SheppLogan13_deformed', I1)


#%% Generate data
name_list = [ 'temporal__t_' + str(i) for i in range(nb_data_points + 1)]
#num_angles_list = [10, 50, 100, 20]
num_angles_list = [20, 30, 50]
maxiangle_list = ['pi']
max_angle_list = [np.pi]
#maxiangle_list = ['pi', '0_25pi', '0_5pi', '0_75pi']
#max_angle_list = [np.pi, 0.25*np.pi, 0.5*np.pi, 0.75*np.pi]
#maxiangle_list = ['0_75pi']
#max_angle_list = [0.75*np.pi]
noise_level_list = [0.0, 0.05, 0.25]
noi_list = ['0', '0_05', '0_25']
#min_angle = 0.25*np.pi
#miniangle = '0_25pi'
min_angle = 0.
miniangle = '0'

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
                        name = name_ground_truth + 'num_angles_' + str(num_angles) + '_min_angle_' + miniangle + '_max_angle_'
                        name += maxiangle + '_noise_' + noi

                        noise = noise_level * odl.phantom.noise.white_noise(forward_op.range)

                        data = forward_op(ground_truth) + noise

                        #data.show()

                        np.savetxt(name, data)

#

#%% Generate data
name_list = ['SheppLogan11_deformed']
#num_angles_list = [10, 50, 100, 20]
num_angles_list = [ 100]
maxiangle_list = ['pi', '0_25pi', '0_5pi', '0_75pi']
max_angle_list = [np.pi, 0.25*np.pi, 0.5*np.pi, 0.75*np.pi]
#maxiangle_list = ['0_75pi']
#max_angle_list = [0.75*np.pi]
noise_level_list = [0.0, 0.05, 0.25]
noi_list = ['0', '0_05', '0_25']
#min_angle = 0.25*np.pi
#miniangle = '0_25pi'
min_angle = 0.
miniangle = '0'

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
                        detector_partition = odl.uniform_partition(-8, 8, int(round(rec_space.shape[0]*np.sqrt(2))))

                        ## Create 2-D parallel projection geometry
                        geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

                        ## Ray transform aka forward projection. We use ASTRA CUDA backend.
                        forward_op = odl.tomo.RayTransform(rec_space, geometry, translation = np.array([-7, 5]), impl='astra_cpu')

                        name_ground_truth = path + name_exp + name_val
                        ground_truth = rec_space.element(np.loadtxt(name_ground_truth))
                        name = name_ground_truth + 'num_angles_' + str(num_angles) + '_min_angle_' + miniangle + '_max_angle_'
                        name += maxiangle + '_noise_' + noi

                        noise = noise_level * odl.phantom.noise.white_noise(forward_op.range)

                        data = forward_op(ground_truth) + noise

                        #data.show()

                        np.savetxt(name, data)


