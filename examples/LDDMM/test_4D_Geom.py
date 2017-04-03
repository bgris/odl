#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 18:04:29 2017

@author: bgris
"""



import numpy as np
import odl
#import copy
#import numpy

##%% Create data from lddmm registration
import matplotlib.pyplot as plt

I0name = '/home/bgris/Downloads/pictures/i_highres.png'
I1name = '/home/bgris/Downloads/pictures/c_highres.png'


# Get digital images
I0 = plt.imread(I0name)
I1 =plt.imread(I1name)

I0 = np.rot90(plt.imread(I0name).astype('float'), -1)[::2, ::2]
I1 = np.rot90(plt.imread(I1name).astype('float'), -1)[::2, ::2]


space = odl.uniform_discr(
    min_pt=[-16, -16], max_pt=[16, 16], shape=[128, 128],
    dtype='float32', interp='linear')

I0=space.element(I0)
I1=space.element(I1)
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


# Maximum iteration number
niter = 20

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
time_itvs = 10

# Give kernel function
def kernel(x):
    sigma = 2.0
    scaled = [xi ** 2 / (2 * sigma ** 2) for xi in x]
    return np.exp(-sum(scaled))
#

#%% Compute by LDDMM solver
image_N0, E, vector_fields = odl.deform.LDDMM_gradient_descent_solver(forward_op, proj_data, I0,
                                            time_itvs, niter, eps, lamb,
                                            kernel, impl, callback)

#%% PLot results


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


image_N0.show(cmap='bone')

initial_grid = vector_fields[0][0].space.points().T
grid = compute_grid_deformation(vector_fields, 1. / time_itvs, initial_grid).reshape(2, 128, 128)
plot_grid(grid, skip=5)
#%%
template=I0
nb_time_point_int=20

#
#data_time_points=np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
##data_time_points=np.array([0,0.2,0.4,0.6,0.8,0.9,1])
#data_space=odl.ProductSpace(forward_op.range,data_time_points.size)
#data=data_space.element([forward_op(image_N0[0]),forward_op(image_N0[5]),
#              forward_op(image_N0[10]),forward_op(image_N0[5]),
#              forward_op(image_N0[0]),forward_op(image_N0[5]),
#              forward_op(image_N0[10]),forward_op(image_N0[5]),
#              forward_op(image_N0[0]),forward_op(image_N0[5]),
#              forward_op(image_N0[10])])
#forward_operators=[forward_op,forward_op,forward_op,forward_op,
#                   forward_op,forward_op,forward_op,forward_op,
#                   forward_op,forward_op,forward_op]
#


#data_time_points=np.array([0,0.5,0.8,1])
data_time_points=np.array([0,0.2,0.4,0.6,0.8,1])
data_space=odl.ProductSpace(forward_op.range,data_time_points.size)
data=data_space.element([forward_op(image_N0[0]),forward_op(image_N0[10]),
              forward_op(image_N0[0]),forward_op(image_N0[10]),
              forward_op(image_N0[0]),forward_op(image_N0[10])])
#data=data_space.element([forward_op(image_N0[0]),forward_op(image_N0[5]),
#              forward_op(image_N0[8]),forward_op(image_N0[10])])

forward_operators=[forward_op,forward_op,forward_op,forward_op,
                   forward_op, forward_op, forward_op]
#data_image=[(image_N0[0]),(image_N0[1]),
#              (image_N0[2]),(image_N0[4]),
#              (image_N0[6]),(image_N0[8]),
#              (image_N0[10])]


Norm=odl.solvers.L2NormSquared(forward_op.range)

energy_op=odl.deform.TemporalAttachmentLDDMMGeom(nb_time_point_int, template ,data,
                            data_time_points, forward_operators,Norm, kernel,
                            domain=None)
#energy=energy_op(vector_fields)

Reg=odl.deform.RegularityLDDMM(kernel,energy_op.domain)
#Reg=odl.deform.RegularityGrowth(kernel,energy_op.domain)
##%%
#grad=energy_op.gradient(vector_fields)

#%%
lam= 1e-8

functional = energy_op + lam*Reg
#
#a=functional.gradient(vector_fields)
#
#b=Reg.gradient(vector_fields)
#functional = 10*energy_op
##%%

vector_fields_list_init=energy_op.domain.zero()

#
#A=odl.solvers.steepest_descent(functional, vector_fields_list_init,0.00001, maxiter=30, tol=1e-16,
#projection=None, callback=None)


##%% gradient descent
vector_fields_list=vector_fields_list_init.copy()
#vector_fields_list=vector_fields.copy()

niter=50
eps = 0.01

attachment_term=energy_op(vector_fields_list)
print(" Initial ,  attachment term : {}".format(attachment_term))

for k in range(niter):
    grad=functional.gradient(vector_fields_list)
    vector_fields_list= (vector_fields_list- eps *grad).copy()
    attachment_term=energy_op(vector_fields_list)
    print(" iter : {}  ,  attachment term : {}".format(k,attachment_term))

#

#%%

I=odl.deform.ShootTemplateFromVectorFields(vector_fields_list, template)

for i in range(int(1*nb_time_point_int)+1):
    u=int(i)
    I[u].show('time {}'.format(u))

#
#%%
for i in range(nb_time_point_int+1):
    ((I[i]-image_N0[i])**2).show()

#%%

data_time_points_index=np.array([0,3,5,7,8,9,10])
for k in range(data_time_points.size):
    ((I[data_time_points_index[k]]-data_image[k])**2).show()

#

#%%

grid = compute_grid_deformation(vector_fields_list, 1. / nb_time_point_int, initial_grid).reshape(2, 128, 128)
plot_grid(grid, skip=5)