#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 15:08:30 2017

@author: bgris
"""





import numpy as np
import odl
#import copy
#import numpy

import matplotlib.pyplot as plt


def compute_grid_deformation(vector_fields_list, time_step, initial_grid):
    vector_fields_list = vector_fields_list
    nb_time_points = vector_fields_list.size

    grid_points = initial_grid

    for t in range(nb_time_points):
        velocity = np.empty_like(grid_points)
        for i, vi in enumerate(vector_fields_list[t]):
            velocity[i, ...] = vi.interpolation(grid_points)
        grid_points += time_step*velocity

    return grid_points


def plot_grid(grid, skip):
    for i in range(0, grid.shape[1], skip):
        plt.plot(grid[0, i, :], grid[1, i, :], 'r', linewidth=0.5)

    for i in range(0, grid.shape[2], skip):
        plt.plot(grid[0, :, i], grid[1, :, i], 'r', linewidth=0.5)

# --- Reading data --- #

# Get the path of data
directory = 'home/bgris/Downloads/pictures/'
I0name = '/home/bgris/Downloads/pictures/i_highres.png'
I1name = '/home/bgris/Downloads/pictures/c_highres.png'



# Get digital images
I0 = plt.imread(I0name)
I1 =plt.imread(I1name)

I0 = np.rot90(plt.imread(I0name).astype('float'), -1)[::2, ::2]
I1 = np.rot90(plt.imread(I1name).astype('float'), -1)[::2, ::2]
# Discrete reconstruction space: discretized functions on the rectangle
space = odl.uniform_discr(
    min_pt=[-16, -16], max_pt=[16, 16], shape=[128, 128],
    dtype='float32', interp='linear')

# Give the number of directions
num_angles = 2

# Create the uniformly distributed directions
angle_partition = odl.uniform_partition(0, np.pi, num_angles,
                                        nodes_on_bdry=[(True, False)])

# Create 2-D projection domain
# The length should be 1.5 times of that of the reconstruction space
detector_partition = odl.uniform_partition(-24, 24, 192)

# Create 2-D parallel projection geometry
geometry = odl.tomo.Parallel2dGeometry(angle_partition,
                                       detector_partition)

# Ray transform aka forward projection. We use ASTRA CUDA backend.
forward_op = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')


# Create projection data by calling the ray transform on the phantom
proj_data = forward_op(I1)

#
#file_path = directory + data_filename
#data, data_extent, header, extended_header = read_mrc_data(file_path=file_path,
#                                                           force_type='FEI1',
#                                                           normalize=True)
#
##Downsample the data
#downsam = 15
#data_downsam = data[:, :, ::downsam]
#
## --- Getting geometry --- #
#
## Create 3-D parallel projection geometry
#single_axis_geometry = geometry_mrc_data(data_extent=data_extent,
#                                         data_shape=data.shape,
#                                         extended_header=extended_header,
#                                         downsam=downsam)
#
## --- Creating reconstruction space --- #
#
## Voxels in 3D region of interest
#rec_shape = (128, 128, 128)
#
## Create reconstruction extent
#rec_extent = np.asarray((1024, 1024, 1024), float)
## Reconstruction space
#
#rec_space = uniform_discr(-rec_extent / 2, rec_extent / 2, rec_shape,
#                          dtype='float32', interp='linear')
#
## --- Creating forward operator --- #
#
## Create forward operator
#forward_op = RayTransform(rec_space, single_axis_geometry, impl='astra_cuda')
#
## --- Chaging the axises of the 3D data --- #
#
## Change the axises of the 3D data
#data_temp1 = np.swapaxes(data_downsam, 0, 2)
#data_temp2 = np.swapaxes(data_temp1, 1, 2)
#data_elem = forward_op.range.element(data_temp2)
## Show one sinograph
#data_elem.show(title='Data in one projection',
#               indices=np.s_[data_elem.shape[0] // 2, :, :])
#

# Maximum iteration number
niter = 10

# Implementation method for mass preserving or not,
# impl chooses 'mp' or 'geom', 'mp' means mass-preserving deformation method,
# 'geom' means geometric deformation method
impl = 'geom'

# Show intermiddle results
callback = odl.solvers.util.callback.CallbackShow(
    '{!r} iterates'.format(impl), display_step=5) & odl.solvers.util.callback.CallbackPrintIteration()

# Give step size for solver
eps = 1e-2

# Give regularization parameter
lamb = 1e-7

# Give the number of time points
time_itvs = 20

# Give kernel function
def kernel(x):
    sigma = 2.0
    scaled = [xi ** 2 / (2 * sigma ** 2) for xi in x]
    return np.exp(-sum(scaled))

# Compute by LDDMM solver
image_N0, E, vector_fields = odl.deform.LDDMM_gradient_descent_solver(forward_op, proj_data, I0,
                                            time_itvs, niter, eps, lamb,
                                            kernel, impl, callback)

image_N0.show(cmap='bone')

initial_grid = vector_fields[0][0].space.points().T
grid = compute_grid_deformation(vector_fields, 1. / time_itvs, initial_grid).reshape(2, 128, 128)
plot_grid(grid, skip=5)
#%%
rec_result_1 = rec_space.element(image_N0[time_itvs // 3])
rec_result_2 = rec_space.element(image_N0[time_itvs * 2 // 3])
rec_result = rec_space.element(image_N0[time_itvs])


# --- Saving reconstructed result --- #


result_2_nii_format(result=rec_result, file_name='triangle_LDDMMrecon3.nii')
result_2_mrc_format(result=rec_result, file_name='triangle_LDDMMrecon3.mrc')


# --- Showing reconstructed result --- #


# Plot the results of interest
plt.figure(1, figsize=(21, 21))
plt.clf()

plt.subplot(2, 2, 1)
plt.imshow(np.asarray(template)[template.shape[0] // 2, :, :], cmap='bone',
           vmin=np.asarray(template).min(),
           vmax=np.asarray(template).max())
plt.colorbar()
plt.title('time_pts = {!r}'.format(0))

plt.subplot(2, 2, 2)
plt.imshow(np.asarray(rec_result_1)[rec_result_1.shape[0] // 2, :, :],
           cmap='bone',
           vmin=np.asarray(template).min(),
           vmax=np.asarray(template).max())
plt.colorbar()
plt.title('time_pts = {!r}'.format(time_itvs // 3))

plt.subplot(2, 2, 3)
plt.imshow(np.asarray(rec_result_2)[rec_result_2.shape[0] // 2, :, :],
           cmap='bone',
           vmin=np.asarray(template).min(),
           vmax=np.asarray(template).max())
plt.colorbar()
plt.title('time_pts = {!r}'.format(time_itvs // 3 * 2))

plt.subplot(2, 2, 4)
plt.imshow(np.asarray(rec_result)[rec_result.shape[0] // 2, :, :],
           cmap='bone',
           vmin=np.asarray(template).min(),
           vmax=np.asarray(template).max())
plt.colorbar()
plt.title('Reconstructed image by {!r} iters, '
    '{!r} projs'.format(niter, single_axis_geometry.partition.shape[0]))

plt.figure(2, figsize=(8, 1.5))
plt.clf()
plt.plot(E)
plt.ylabel('Energy')
# plt.gca().axes.yaxis.set_ticklabels(['0']+['']*8)
plt.gca().axes.yaxis.set_ticklabels([])
plt.grid(True)
