#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 11:31:20 2017

@author: bgris
"""

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

"""
ET image reconstruction using LDDMM.
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
import numpy as np
import matplotlib.pyplot as plt
import odl
from odl.discr import uniform_discr, Gradient
from odl.phantom import sphere, sphere2, cube


import odl.deform.mrc_data_io

from odl.deform.mrc_data_io import (read_mrc_data, geometry_mrc_data,
                                    result_2_mrc_format, result_2_nii_format)
from odl.tomo import RayTransform, fbp_op
from odl.operator import (BroadcastOperator, power_method_opnorm)
from odl.solvers import (CallbackShow, CallbackPrintIteration, ZeroFunctional,
                         L2NormSquared, L1Norm, SeparableSum,
                         chambolle_pock_solver, conjugate_gradient_normal)
standard_library.install_aliases()


#%%%
# --- Reading data --- #

# Get the path of data radon
#directory = '/home/bgris/data/TEM/wetransfer-569840/'

# Get the path of data cormack
directory = '/home/bgris/data/TEM/wetransfer-569840/'

data_filename = 'triangle_crop.mrc'
file_path = directory + data_filename
data, data_extent, header, extended_header = read_mrc_data(file_path=file_path,
                                                           force_type='FEI1',
                                                           normalize=True)

#Downsample the data
downsam = 20
data_downsam = data[:, :, ::downsam]

# --- Getting geometry --- #
det_pix_size = 0.521

# Create 3-D parallel projection geometry
single_axis_geometry = odl.deform.mrc_data_io.geometry_mrc_data(data_extent=data_extent,
                                         data_shape=data.shape, det_pix_size=det_pix_size,
                                         units='physical',
                                         extended_header=extended_header,
                                         downsam=downsam)


# --- Creating reconstruction space --- #

# Voxels in 3D region of interest
rec_shape = (data.shape[0], data.shape[0], data.shape[0])

## Create reconstruction extent
## for rod
#min_pt = np.asarray((-150, -150, -150), float)
#max_pt = np.asarray((150, 150, 150), float)

# Create reconstruction extent
# for triangle, sphere
rec_extent = np.asarray(rec_shape, float)
#min_pt = np.asarray((-100, -100, -100), float)
#max_pt = np.asarray((100, 100, 100), float)

# Reconstruction space with physical setting
rec_space = uniform_discr(-rec_extent / 2 * det_pix_size,
                          rec_extent / 2  * det_pix_size, rec_shape,
dtype='float32', interp='linear')

# --- Creating forward operator --- #

# Create forward operator
forward_op = RayTransform(rec_space, single_axis_geometry, impl='astra_cuda')

# --- Chaging the axises of the 3D data --- #

# Change the axises of the 3D data
#data = np.where(data >= 30, data, 0.0)
data_temp1 = np.swapaxes(data_downsam, 0, 2)
data_temp2 = np.swapaxes(data_temp1, 1, 2)
data_elem = forward_op.range.element(data_temp2)

# Show one sinograph
#data_elem.show(title='Data in one projection',
#               indices=np.s_[data_elem.shape[0] // 2, :, :])



data_space = uniform_discr(-rec_extent / 2 * det_pix_size,
                          rec_extent / 2  * det_pix_size, (200,200,151),
dtype='float32', interp='linear')


## sphere for rod, triangle, sphere2 for sphere
#template = sphere(rec_space, smooth=True, taper=50.0, radius=30)
#%%


import nibabel as nib
import os
lam_fbp=0.8
# BUG ?: does not work with padding=True
fbp = odl.tomo.fbp_op(forward_op, padding=False, filter_type='Hann', frequency_scaling=lam_fbp)

reco_fbp=fbp(data_elem)
A = np.asarray(reco_fbp)
img = nib.Nifti1Image(A, np.eye(4))
img.get_data_dtype() == np.dtype(np.float32)
img.header.get_xyzt_units()
img.to_filename(os.path.join('/home/bgris/Results/Metamorphosis/TEM/','reco_fbp_'+ str(lam_fbp) + '.nii'))

template=reco_fbp.copy()


#%% Modifying template

# step template with 1/0 inside/outsi
template_step=1.0*np.array((np.asarray(template)>0.5)).copy()

template=rec_space.element(template_step).copy()

#%%
#mean = np.sum(data_temp2[data_temp2.shape[0] // 2]) / data_temp2[data_temp2.shape[0] // 2].size
#temp = data_temp2[data_temp2.shape[0] // 2] - mean
#temp = np.where(temp >= 0.01, data_temp2[data_temp2.shape[0] // 2], 0.0)
#mass_from_data = np.sum(temp)
#
## Evaluate the mass of template
#mass_template = np.sum(np.asarray(template))
#
## Get the same mass for template
#template = mass_from_data / mass_template * template
#
#template_min=np.asarray(template).min()
#data_min=data_temp2.min()
#
#template=rec_space.element(np.asarray(template)-template_min)
#data_elem= forward_op.range.element(data_temp2-data_min)

##%%

# Maximum iteration number
niter = 300

# Give step size for solver
eps = 0.01

# Give regularization parameter
lamb = 1*1e-3
tau = 1* 1e-3
# Give the number of time points
time_itvs = 5
nb_time_point_int=time_itvs
# Choose parameter for kernel function
sigma = 3

# Give kernel function
def kernel(x):
    scaled = [xi ** 2 / (2 * sigma ** 2) for xi in x]
    return np.exp(-sum(scaled))



# Initialize the reconstruction result
rec_result = template
data_list=[forward_op.range.element(data_elem)]
data_time_points=[1]
forward_operators=[forward_op]
Norm=odl.solvers.L2NormSquared(forward_op.range)

#
#forward_op(template).show(indices=np.s_[data_elem.shape[0] // 2, :, :])
#data_elem.show(title='Data in one projection',
#               indices=np.s_[data_elem.shape[0] // 2, :, :])
#
#
#
#forward_op(template).show(indices=np.s_[:,data_elem.shape[1] // 2,  :])
#data_elem.show(title='Data in one projection',
#               indices=np.s_[:,data_elem.shape[1] // 2,  :])
#
#
#
#forward_op(template).show(indices=np.s_[:,:,data_elem.shape[2] // 2])
#data_elem.show(title='Data in one projection',
#               indices=np.s_[:,:,data_elem.shape[2] // 2])
#li.append(Norm*(forward_operators[0] - data_list[0]))

##%% Define energy operator
#energy_op=odl.deform.TemporalAttachmentLDDMMGeom(nb_time_point_int, template ,data_list,
#                            data_time_points, forward_operators,Norm, kernel,
#                            domain=None)
#
#
#Reg=odl.deform.RegularityLDDMM(kernel,energy_op.domain)
#
#functional = energy_op + lamb*Reg


#%%

functional=odl.deform.TemporalAttachmentMetamorphosisGeom(nb_time_point_int,
                            lamb,tau,template ,data_list,
                            data_time_points, forward_operators,Norm, kernel,
                            domain=None)

grad_op=functional.gradient

#%% Gradient descent

X_init=functional.domain.zero()
X=X_init.copy()
#%%

#nameinit='   OlivettiFaces_smoothing_1_directmatching'
name0= 'ET_Metamorphosis_sigma_3_lam_1_e__3_tau_1_e__3_nbInt_5_iter_300_initialisation_fbp_steped_directions_151_registration_8'

#%%
import nibabel as nib
import os

#attachment_term=functional(X)
#print(" Initial ,  attachment term : {}".format(attachment_term))
epsV=0.00002
epsZ=0.00002
energy=functional(X)
print(" Initial ,  energy : {}".format(energy))
cont=0
for k in range(niter):
    #for t in range(nb_time_point_int):
    #   np.savetxt('vector_fields_list' + str(t),np.asarray(vector_fields_list[t]))
    if cont==0:
        grad=grad_op(X)
        #grad=functional.gradient(X)
    #X[0]= (X[0]- epsV *grad[0]).copy()
    #X[1]= (X[1]- epsZ *grad[1]).copy()
    X_temp0=X.copy()
    X_temp0[0]= (X[0]- epsV *grad[0]).copy()
    X_temp0[1]= (X[1]- epsZ *grad[1]).copy()
    energy_temp0=functional(X_temp0)
    if energy_temp0<energy:
        X=X_temp0.copy()
        energy=energy_temp0
        epsV*=1.5
        epsZ*=1.5
        cont=0
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
            epsV*=1.2
            epsZ*=1.2
            cont=0
            print(" iter : {}  , energy : {}, epsV = {} , epsZ = {}".format(k,energy,epsV, epsZ))
        else:
            epsV*=0.5
            epsZ*=0.5
            cont=1
            print("epsV = {} , epsZ = {}".format(epsV, epsZ))


    if(k%5 == 0):
        image_list_data=functional.ComputeMetamorphosis(X[0],X[1])
        template_evo=odl.deform.ShootTemplateFromVectorFields(X[0], template)
        zeta_transp=odl.deform.ShootSourceTermBackwardlist(X[0],X[1])
        image_evol=odl.deform.IntegrateTemplateEvol(template,zeta_transp,0,nb_time_point_int)

        img = nib.Nifti1Image(image_list_data[0], np.eye(4))
        img.get_data_dtype() == np.dtype(np.float32)
        img.header.get_xyzt_units()
        name = name0 + '_metamorphosis' + str(k) + '.nii'
        img.to_filename(os.path.join('/home/bgris/Results/Metamorphosis/TEM/temp/',name))

        img = nib.Nifti1Image(template_evo[nb_time_point_int], np.eye(4))
        img.get_data_dtype() == np.dtype(np.float32)
        img.header.get_xyzt_units()
        name = name0 + '_template_evo' + str(k) + '.nii'
        img.to_filename(os.path.join('/home/bgris/Results/Metamorphosis/TEM/temp/',name))

        img = nib.Nifti1Image(image_evol[nb_time_point_int], np.eye(4))
        img.get_data_dtype() == np.dtype(np.float32)
        img.header.get_xyzt_units()
        name = name0 + '_image_evo' + str(k) + '.nii'
        img.to_filename(os.path.join('/home/bgris/Results/Metamorphosis/TEM/temp/',name))

#

#%% Compute estimated trajectory
image_list_data=functional.ComputeMetamorphosis(X[0],X[1])

#image_list_data[0].show()
#image_list_data[0].show(clim=[0,1])

image_list=functional.ComputeMetamorphosisListInt(X[0],X[1])

#for i in range(nb_time_point_int+1):
#    #image_list[i].show('Metamorphosis time {}'.format(i))
#    #image_list[i].show('Metamorphosis time {}'.format(i),clim=[0,1])
#    image_list[i].show('Metamorphosis time {}'.format(i),clim=[-0.2,1.2])


template_evo=odl.deform.ShootTemplateFromVectorFields(X[0], template)

#for i in range(nb_time_point_int+1):
#    template_evo[i].show('Template evolution time {} '.format(i))
#

zeta_transp=odl.deform.ShootSourceTermBackwardlist(X[0],X[1])
#template_evolution=odl.deform.IntegrateTemplateEvol(functional.template,zeta_transp,0,functional.N)

image_evol=odl.deform.IntegrateTemplateEvol(template,zeta_transp,0,nb_time_point_int)
#for i in range(nb_time_point_int+1):
#    image_evol[i].show('Image evolution time {} '.format(i),clim=[1,1.1])


#
#grid_points=compute_grid_deformation_list(X[0], 1/nb_time_point_int, template.space.points().T)

#
#for t in range(nb_time_point_int):
#    grid=grid_points[t].reshape(2, 128, 128).copy()
#plot_grid(grid, 2)


#image_N0=odl.deform.ShootTemplateFromVectorFields(vector_fields_list, template)


import nibabel as nib
import os
for t in range(nb_time_point_int+1):
    # Save reconstruction
    data = image_list[t].asarray().copy()
    img = nib.Nifti1Image(data, np.eye(4))
    img.get_data_dtype() == np.dtype(np.float32)
    img.header.get_xyzt_units()
    name = name0 + '_Metamorphosis_t_' + str(t) + '.nii'
    img.to_filename(os.path.join('/home/bgris/Results/Metamorphosis/TEM/',name))

    # Save reconstruction
    data = template_evo[t].asarray().copy()
    img = nib.Nifti1Image(data, np.eye(4))
    img.get_data_dtype() == np.dtype(np.float32)
    img.header.get_xyzt_units()
    name = name0 + '_template_evo_t_' + str(t) + '.nii'
    img.to_filename(os.path.join('/home/bgris/Results/Metamorphosis/TEM/',name))

    # Save reconstruction
    data = image_evol[t].asarray().copy()
    img = nib.Nifti1Image(data, np.eye(4))
    img.get_data_dtype() == np.dtype(np.float32)
    img.header.get_xyzt_units()
    name = name0 + '_image_evo_t_' + str(t) + '.nii'
    img.to_filename(os.path.join('/home/bgris/Results/Metamorphosis/TEM/',name))



#
#
#image_N0[0].show(indices=np.s_[rec_shape[0] // 2, :, :])
#image_N0[20].show(indices=np.s_[rec_shape[0] // 2, :, :])
#
#
#forward_op(image_N0[20]).show(indices=np.s_[:,:,data_elem.shape[2] // 2])
#data_elem.show(indices=np.s_[:,:,data_elem.shape[2] // 2])
#
#
#
#odl.deform.mrc_data_io.result_2_mrc_format(result=image_N0[nb_time_point_int],
#                    file_name='Reconstruction3D_ET.mrc')
#
#odl.deform.mrc_data_io.result_2_mrc_format(result=template,
#                    file_name='template_ET.mrc')



if False:
    data = template.asarray().copy()
    img = nib.Nifti1Image(data, np.eye(4))
    img.get_data_dtype() == np.dtype(np.float32)
    img.header.get_xyzt_units()
    name = 'template_ET'  + '.nii'
    img.to_filename(os.path.join('/home/bgris/Results/Metamorphosis/TEM/temp/',name))


#%% Gradient descent only on the deformation part

name0 +='relaunched_onle_defo'
import nibabel as nib
import os

#attachment_term=functional(X)
#print(" Initial ,  attachment term : {}".format(attachment_term))
epsV=0.00002
epsZ=0.00002
energy=functional(X)
print(" Initial ,  energy : {}".format(energy))
cont=0
for k in range(niter):
    #for t in range(nb_time_point_int):
    #   np.savetxt('vector_fields_list' + str(t),np.asarray(vector_fields_list[t]))
    if cont==0:
        grad=grad_op(X)
        #grad=functional.gradient(X)
    #X[0]= (X[0]- epsV *grad[0]).copy()
    #X[1]= (X[1]- epsZ *grad[1]).copy()
    X_temp0=X.copy()
    X_temp0[0]= (X[0]- epsV *grad[0]).copy()
    #X_temp0[1]= (X[1]- epsZ *grad[1]).copy()
    energy_temp0=functional(X_temp0)
    if energy_temp0<energy:
        X=X_temp0.copy()
        energy=energy_temp0
        epsV*=1.5
        #epsZ*=1.5
        cont=0
        print(" iter : {}  , energy : {}, epsV = {} , epsZ = {}".format(k,energy,epsV, epsZ))
        """else:
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
            epsV*=1.2
            epsZ*=1.2
            cont=0
            print(" iter : {}  , energy : {}, epsV = {} , epsZ = {}".format(k,energy,epsV, epsZ))

        else:
            epsV*=0.5
            epsZ*=0.5
            cont=1
            print("epsV = {} , epsZ = {}".format(epsV, epsZ))
        """
    else:
        epsV*=0.5
        #epsZ*=0.5
        cont=1
        print("epsV = {}".format(epsV))


    if(k%5 == 0):
        image_list_data=functional.ComputeMetamorphosis(X[0],X[1])
        template_evo=odl.deform.ShootTemplateFromVectorFields(X[0], template)
        zeta_transp=odl.deform.ShootSourceTermBackwardlist(X[0],X[1])
        image_evol=odl.deform.IntegrateTemplateEvol(template,zeta_transp,0,nb_time_point_int)

        img = nib.Nifti1Image(image_list_data[0], np.eye(4))
        img.get_data_dtype() == np.dtype(np.float32)
        img.header.get_xyzt_units()
        name = name0 + '_metamorphosis' + str(k) + '.nii'
        img.to_filename(os.path.join('/home/bgris/Results/Metamorphosis/TEM/temp/',name))

        img = nib.Nifti1Image(template_evo[nb_time_point_int], np.eye(4))
        img.get_data_dtype() == np.dtype(np.float32)
        img.header.get_xyzt_units()
        name = name0 + '_template_evo' + str(k) + '.nii'
        img.to_filename(os.path.join('/home/bgris/Results/Metamorphosis/TEM/temp/',name))

        img = nib.Nifti1Image(image_evol[nb_time_point_int], np.eye(4))
        img.get_data_dtype() == np.dtype(np.float32)
        img.header.get_xyzt_units()
        name = name0 + '_image_evo' + str(k) + '.nii'
        img.to_filename(os.path.join('/home/bgris/Results/Metamorphosis/TEM/temp/',name))

#



#%% fbp

import nibabel as nib
import os
lam_fbp=0.8
fbp = odl.tomo.fbp_op(forward_op, filter_type='Hann', frequency_scaling=lam_fbp)

reco_fbp=fbp(data_elem)
A = np.asarray(reco_fbp)
img = nib.Nifti1Image(A, np.eye(4))
img.get_data_dtype() == np.dtype(np.float32)
img.header.get_xyzt_units()
img.to_filename(os.path.join('/home/bgris/Results/Metamorphosis/TEM/','reco_fbp_'+ str(lam_fbp) + '.nii'))



























#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 17:18:08 2017

@author: bgris
"""

