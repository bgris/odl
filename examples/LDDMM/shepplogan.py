#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 18:03:37 2017

@author: bgris
"""


import odl
from odl.deform.linearized import _linear_deform
import numpy as np
from matplotlib import pylab as plt



def snr_fun(signal, noise, impl):
    """Compute the signal-to-noise ratio.
    Parameters
    ----------
    signal : `array-like`
        Noiseless data.
    noise : `array-like`
        Noise.
    impl : {'general', 'dB'}
        Implementation method.
        'general' means SNR = variance(signal) / variance(noise),
        'dB' means SNR = 10 * log10 (variance(signal) / variance(noise)).
    Returns
    -------
    snr : `float`
        Value of signal-to-noise ratio.
        If the power of noise is zero, then the return is 'inf',
        otherwise, the computed value.
    """
    if np.abs(np.asarray(noise)).sum() != 0:
        ave1 = np.sum(signal) / signal.size
        ave2 = np.sum(noise) / noise.size
        s_power = np.sqrt(np.sum((signal - ave1) * (signal - ave1)))
        n_power = np.sqrt(np.sum((noise - ave2) * (noise - ave2)))
        if impl == 'general':
            return s_power / n_power
        elif impl == 'dB':
            return 10.0 * np.log10(s_power / n_power)
        else:
            raise ValueError('unknown `impl` {}'.format(impl))
    else:
        return float('inf')




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


def compute_grid_deformation_list(vector_fields_list, time_step, initial_grid):
    vector_fields_list = vector_fields_list
    nb_time_points = vector_fields_list.size
    grid_list=[]
    grid_points=initial_grid.copy()
    grid_list.append(initial_grid)

    for t in range(nb_time_points):
        velocity = np.empty_like(grid_points)
        for i, vi in enumerate(vector_fields_list[t]):
            velocity[i, ...] = vi.interpolation(grid_points)
        grid_points += time_step*velocity
        grid_list.append(grid_points.copy())

    return grid_list


# As previously but check if points of the grids are in the
# Domain and if they are not the velocity is equal to zero
def compute_grid_deformation_list_bis(vector_fields_list, time_step, initial_grid):
    vector_fields_list = vector_fields_list
    nb_time_points = vector_fields_list.size
    grid_list=[]
    grid_points=initial_grid.T.copy()
    grid_list.append(initial_grid.T)
    mini=vector_fields_list[0].space[0].min_pt
    maxi=vector_fields_list[0].space[0].max_pt
    for t in range(nb_time_points):
        print(t)
        velocity = np.zeros_like(grid_points)
        for i, vi in enumerate(vector_fields_list[t]):
            for k in range(len(initial_grid.T)):
                isindomain=1
                point=grid_points[k]
                for u in range(len(mini)):
                    if (point[u]<mini[u] or point[u]>maxi[u] ):
                        isindomain=0
                if (isindomain==1):
                    velocity[k][i] = vi.interpolation(point)

        grid_points += time_step*velocity
        grid_list.append(grid_points.copy().T)

    return grid_list


def plot_grid(grid, skip):
    for i in range(0, grid.shape[1], skip):
        plt.plot(grid[0, i, :], grid[1, i, :], 'r', linewidth=0.5)

    for i in range(0, grid.shape[2], skip):
        plt.plot(grid[0, :, i], grid[1, :, i], 'r', linewidth=0.5)



#
#%%# Discrete reconstruction space: discretized functions on the rectangle

rec_space = odl.uniform_discr(
    min_pt=[-16,-16], max_pt=[16,16], shape=[256, 256],
    dtype='float32', interp='linear')

# Create the template as the deformed Shepp-Logan phantom
ground_truth = odl.phantom.transmission.shepp_logan(rec_space, modified=True)

# Create the template for Shepp-Logan phantom
deform_field_space = rec_space.tangent_bundle
disp_func = [
    lambda x: 16.0 * np.sin(np.pi * x[0] / 40.0),
    lambda x: 16.0 * np.sin(np.pi * x[1] / 36.0)]
deform_field = deform_field_space.element(disp_func)

template = rec_space.element(
        odl.deform.mass_preserving.geometric_deform(odl.phantom.transmission.shepp_logan(rec_space, modified=True), deform_field))

# Implementation method for mass preserving or not,
# impl chooses 'mp' or 'geom', 'mp' means mass-preserving deformation method,
# 'geom' means geometric deformation method
impl1 = 'geom'



#ground_truth.show('ground truth')
#template.show('template')

# The parameter for kernel function
sigma = 2.0

# Give kernel function
def kernel(x):
    scaled = [xi ** 2 / (2 * sigma ** 2) for xi in x]
    return np.exp(-sum(scaled))

# Maximum iteration number
niter = 500

# Give step size for solver
eps = 0.02

# Give regularization parameter
lamb = 1e-7

# Give the number of directions
num_angles = 10

# Create the uniformly distributed directions
angle_partition = odl.uniform_partition(0.0, np.pi, num_angles,
                                    nodes_on_bdry=[(True, True)])

# Create 2-D projection domain
# The length should be 1.5 times of that of the reconstruction space
detector_partition = odl.uniform_partition(-24, 24, 620)

# Create 2-D parallel projection geometry
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

# Ray transform aka forward projection. We use ASTRA CUDA backend.
forward_op = odl.tomo.RayTransform(rec_space, geometry, impl='astra_cpu')

# Create projection data by calling the op on the phantom
proj_data = forward_op(ground_truth)

# Add white Gaussion noise onto the noiseless data
noise = 0.5 * odl.phantom.noise.white_noise(forward_op.range)

## Create the noisy projection data
noise_proj_data = proj_data + noise

# Compute the signal-to-noise ratio in dB
snr = snr_fun(proj_data, noise_proj_data - proj_data, impl='dB')

# Output the signal-to-noise ratio
print('snr = {!r}'.format(snr))

# Give the number of time points
time_itvs = 20
nb_time_point_int=time_itvs

data=[noise_proj_data]
data_time_points=np.array([1])
forward_operators=[forward_op]
Norm=odl.solvers.L2NormSquared(forward_op.range)
#%% Define energy operator
energy_op=odl.deform.TemporalAttachmentLDDMMGeom(nb_time_point_int, template ,data,
                            data_time_points, forward_operators,Norm, kernel,
                            domain=None)

Reg=odl.deform.RegularityLDDMM(kernel,energy_op.domain)

functional = energy_op + lamb*Reg

#%% Gradient descent

vector_fields_list_init=energy_op.domain.zero()
vector_fields_list=vector_fields_list_init.copy()
attachment_term=energy_op(vector_fields_list)
print(" Initial ,  attachment term : {}".format(attachment_term))

for k in range(niter):
    grad=functional.gradient(vector_fields_list)
    vector_fields_list= (vector_fields_list- eps *grad).copy()
    attachment_term=energy_op(vector_fields_list)
    print(" iter : {}  ,  attachment term : {}".format(k,attachment_term))
#
#%% Compute estimated trajectory
image_N0=odl.deform.ShootTemplateFromVectorFields(vector_fields_list, template)
#
grid_points=compute_grid_deformation_list(vector_fields_list, 1/nb_time_point_int, template.space.points().T)

#%%
t=nb_time_point_int
image_N0[t].show('Mixed strategy time {}'.format(t+1))
grid=grid_points[t].reshape(2, rec_space.shape[0], rec_space.shape[1]).copy()
plot_grid(grid, 5)
#
#%%
rec_result_1 = rec_space.element(image_N0[time_itvs // 4])
rec_result_2 = rec_space.element(image_N0[time_itvs // 4 * 2])
rec_result_3 = rec_space.element(image_N0[time_itvs // 4 * 3])
rec_result = rec_space.element(image_N0[time_itvs])

# Compute the projections of the reconstructed image
rec_proj_data = forward_op(rec_result)

#%%%
# Plot the results of interest
plt.figure(1, figsize=(24, 24))
#plt.clf()

plt.subplot(3, 3, 1)
plt.imshow(np.rot90(template), cmap='bone',
           vmin=np.asarray(template).min(),
           vmax=np.asarray(template).max())
plt.axis('off')
#plt.savefig("/home/chchen/SwedenWork_Chong/NumericalResults_S/LDDMM_results/J_V/template_J.png", bbox_inches='tight')
plt.colorbar()
plt.title('Template')

plt.subplot(3, 3, 2)
plt.imshow(np.rot90(rec_result_1), cmap='bone',
           vmin=np.asarray(rec_result_1).min(),
           vmax=np.asarray(rec_result_1).max())
plt.axis('off')
plt.colorbar()
plt.title('time_pts = {!r}'.format(time_itvs // 4))

plt.subplot(3, 3, 3)
plt.imshow(np.rot90(rec_result_2), cmap='bone',
           vmin=np.asarray(rec_result_2).min(),
           vmax=np.asarray(rec_result_2).max())
plt.axis('off')
plt.colorbar()
plt.title('time_pts = {!r}'.format(time_itvs // 4 * 2))

plt.subplot(3, 3, 4)
plt.imshow(np.rot90(rec_result_3), cmap='bone',
           vmin=np.asarray(rec_result_3).min(),
           vmax=np.asarray(rec_result_3).max())
plt.axis('off')
plt.colorbar()
plt.title('time_pts = {!r}'.format(time_itvs // 4 * 3))

plt.subplot(3, 3, 5)
plt.imshow(np.rot90(rec_result), cmap='bone',
           vmin=np.asarray(rec_result).min(),
           vmax=np.asarray(rec_result).max())
plt.axis('off')
plt.colorbar()
plt.title('Reconstructed by {!r} iters, '
    '{!r} projs'.format(niter, num_angles))

plt.subplot(3, 3, 6)
plt.imshow(np.rot90(ground_truth), cmap='bone',
           vmin=np.asarray(ground_truth).min(),
           vmax=np.asarray(ground_truth).max())
plt.axis('off')
plt.colorbar()
plt.title('Ground truth')


name=nameinit + 'LDDMM.png'
plt.savefig(name, bbox_inches='tight')

