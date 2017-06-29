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

# Get the path of data
directory = '/home/bgris/odl/examples/TEM/wetransfer-569840/'
data_filename = 'triangle_crop.mrc'
file_path = directory + data_filename
data, data_extent, header, extended_header = read_mrc_data(file_path=file_path,
                                                           force_type='FEI1',
                                                           normalize=True)

#Downsample the data
downsam = 1
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
template = sphere(rec_space, smooth=True, taper=10.0, radius=0.5)
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

#%%

# Maximum iteration number
niter = 150

# Give step size for solver
eps = 0.01

# Give regularization parameter
lamb = 1*1e-11
tau = 100* 1e0
# Give the number of time points
time_itvs = 5
nb_time_point_int=time_itvs
# Choose parameter for kernel function
sigma = 0.5

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

#%% Define energy operator
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
import nibabel as nib
import os

attachment_term=functional(X)
print(" Initial ,  attachment term : {}".format(attachment_term))
epsV=0.02
epsZ=0.0002
for k in range(niter):
    #for t in range(nb_time_point_int):
    #   np.savetxt('vector_fields_list' + str(t),np.asarray(vector_fields_list[t]))
    grad=grad_op(X)
    X[0]= (X[0]- epsV *grad[0]).copy()
    X[1]= (X[1]- epsZ *grad[1]).copy()
    ener=functional(X)
    print(" iter : {}  ,  Energy : {}".format(k,ener))
    if(k%5 == 0):
        image_list_data=functional.ComputeMetamorphosis(X[0],X[1])
        template_evo=odl.deform.ShootTemplateFromVectorFields(X[0], template)
        zeta_transp=odl.deform.ShootSourceTermBackwardlist(X[0],X[1])
        image_evol=odl.deform.IntegrateTemplateEvol(template,zeta_transp,0,nb_time_point_int)

        img = nib.Nifti1Image(image_list_data[0], np.eye(4))
        img.get_data_dtype() == np.dtype(np.float32)
        img.header.get_xyzt_units()
        name = 'ET_metamorphosis' + str(k) + '.nii'
        img.to_filename(os.path.join('/home/bgris/odl/examples/Metamorphosis/',name))

        img = nib.Nifti1Image(template_evo[nb_time_point_int], np.eye(4))
        img.get_data_dtype() == np.dtype(np.float32)
        img.header.get_xyzt_units()
        name = 'ET_template_evo' + str(k) + '.nii'
        img.to_filename(os.path.join('/home/bgris/odl/examples/Metamorphosis/',name))

        img = nib.Nifti1Image(image_evol[nb_time_point_int], np.eye(4))
        img.get_data_dtype() == np.dtype(np.float32)
        img.header.get_xyzt_units()
        name = 'ET_image_evo' + str(k) + '.nii'
        img.to_filename(os.path.join('/home/bgris/odl/examples/Metamorphosis/',name))

#

#%%


image_N0=odl.deform.ShootTemplateFromVectorFields(vector_fields_list, template)


import nibabel as nib
import os
for t in range(nb_time_point_int+1):
    # Save reconstruction
    data = image_N0[t].asarray().copy()
    img = nib.Nifti1Image(data, np.eye(4))
    img.get_data_dtype() == np.dtype(np.float32)
    img.header.get_xyzt_units()
    name = 'ET' + str(t) + '.nii'
    img.to_filename(os.path.join('/home/bgris/odl/examples/TEM/',name))





image_N0[0].show(indices=np.s_[rec_shape[0] // 2, :, :])
image_N0[20].show(indices=np.s_[rec_shape[0] // 2, :, :])


forward_op(image_N0[20]).show(indices=np.s_[:,:,data_elem.shape[2] // 2])
data_elem.show(indices=np.s_[:,:,data_elem.shape[2] // 2])



odl.deform.mrc_data_io.result_2_mrc_format(result=image_N0[nb_time_point_int],
                    file_name='Reconstruction3D_ET.mrc')

odl.deform.mrc_data_io.result_2_mrc_format(result=template,
                    file_name='template_ET.mrc')





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
img.to_filename(os.path.join('/home/bgris/odl/examples/TEM','reco_fbp_'+ str(lam_fbp) + '.nii'))



























#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 17:18:08 2017

@author: bgris
"""

