#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 11:39:31 2017

@author: bgris
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 15:47:58 2017

@author: bgris
"""

import odl
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
#%%

space = odl.uniform_discr(
    min_pt=[-16, -16], max_pt=[16, 16], shape=[512, 512],
    dtype='float32', interp='linear')
path = '/home/bgris/data/SheppLoganRotationSmallDef/Images/'
Name_target = '/home/bgris/data/SheppLoganRotationSmallDef/target_2affine_1rotation_512'
Name_target = '/home/barbara/data/SheppLoganRotationSmallDef/target_2affine_1rotation_512'
Name_target = '/home/barbara/data/SheppLogan/Rotation'
Name_target = path + 'source_5'


template=space.element(np.loadtxt(path + 'source_0'))
target = space.element(np.loadtxt(Name_target))

# Discrete reconstruction space: discretized functions on the rectangle
rec_space = space
# Create the ground truth as the given image
ground_truth = rec_space.element(target)


# Create the template as the given image
template = rec_space.element(template)


# Implementation method for mass preserving or not,
# impl chooses 'mp' or 'geom', 'mp' means mass-preserving deformation method,
# 'geom' means geometric deformation method
impl1 = 'geom'

# Implementation method for least square data matching term
impl2 = 'least_square'

# Show intermiddle results
#callback = CallbackShow(
#    '{!r} {!r} iterates'.format(impl1, impl2), display_step=5) & \
#    CallbackPrintIteration()

ground_truth.show('ground truth')
template.show('template')

# The parameter for kernel function
sigma = 1.0

# Give kernel function
def kernel(x):
    scaled = [xi ** 2 / ( sigma ** 2) for xi in x]
    return np.exp(-sum(scaled))

# Maximum iteration number
niter = 200

# Give step size for solver
eps = 0.02

# Give regularization parameter
lamb = 1e-7

# Give the number of directions
#num_angles = 10

## Create the uniformly distributed directions
#angle_partition = odl.uniform_partition(0.0, np.pi, num_angles,
#                                    nodes_on_bdry=[(True, True)])
#
## Create 2-D projection domain
## The length should be 1.5 times of that of the reconstruction space
#detector_partition = odl.uniform_partition(-24, 24, int(round(rec_space.shape[0]*np.sqrt(2))))
#
## Create 2-D parallel projection geometry
#geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)
#
## Ray transform aka forward projection. We use ASTRA CUDA backend.
#forward_op = odl.tomo.RayTransform(rec_space, geometry, impl='astra_cpu')

forward_op=odl.IdentityOperator(rec_space)

# Create projection data by calling the op on the phantom
proj_data = forward_op(ground_truth)

# Add white Gaussion noise onto the noiseless data
noise =0.05 * odl.phantom.noise.white_noise(forward_op.range)

# Create the noisy projection data
noise_proj_data = proj_data + noise

# Compute the signal-to-noise ratio in dB
snr = snr_fun(proj_data, noise_proj_data - proj_data, impl='dB')

# Output the signal-to-noise ratio
print('snr = {!r}'.format(snr))

# Give the number of time points
time_itvs = 10
nb_time_point_int=time_itvs

data=[proj_data]
#data=[noise_proj_data]
data_time_points=np.array([1])
forward_operators=[forward_op]
Norm=odl.solvers.L2NormSquared(forward_op.range)

#%% Define energy operator
energy_op=odl.deform.TemporalAttachmentLDDMMGeom(nb_time_point_int, template ,data,
                            data_time_points, forward_operators,Norm, kernel,
                            domain=None)


Reg=odl.deform.RegularityLDDMM(kernel,energy_op.domain)

functional = energy_op + lamb*Reg

#%% Gradient descent : initialisation
eps = 0.05
import copy
vector_fields_list_init=energy_op.domain.zero()
#vector_fields_list_init=energy_op.domain.element(copy.deepcopy(vector_fields_list_load))

vector_fields_list=vector_fields_list_init.copy()

attachment_term=energy_op(vector_fields_list)
niter = 100
#%%Gradient descent : descent
print(" Initial ,  attachment term : {}".format(attachment_term))
import copy
for k in range(niter):
    grad=functional.gradient(vector_fields_list)
    vector_fields_list_temp= copy.deepcopy(vector_fields_list- eps *grad)
    attachment_term_temp=energy_op(vector_fields_list_temp)
    if attachment_term_temp < attachment_term:
        attachment_term = attachment_term_temp
        vector_fields_list = copy.deepcopy(vector_fields_list_temp)
        eps *= 1.2
    else:
        eps *= 0.8
    print(" iter : {}  ,  attachment term : {}".format(k,attachment_term))
#
#%% Compute estimated trajectory
#

image_N0=odl.deform.ShootTemplateFromVectorFields(energy_op.domain.element(vector_fields_list), template)



grid_points=compute_grid_deformation_list_bis(vector_fields_list, 1/nb_time_point_int, template.space.points().T)

#
#for t in range(nb_time_point_int):
#    grid=grid_points[t].reshape(2, 128, 128).copy()
#plot_grid(grid, 2)
#%% save figures at each time
name_exp = 'EstimatedTrajectory_SheppLogan_Rotation_LDDMMsigma_1_no_noise'
name = '/home/bgris/Results/LDDMM/SheppLogan/Rotation/' + name_exp

for i in range(nb_time_point_int + 1):

    fig = image_N0[i].show(clim = [0,2])
    plt.axis('off')
    fig.delaxes(fig.axes[1])
    plt.savefig(name + '_' + str(i), box_inches='tight')
#

#%% Def plotting function
mini = 0
maxi = 1
def plot_result(name,image_N0):
    plt.figure()
    rec_result_1 = rec_space.element(image_N0[time_itvs // 4])
    rec_result_2 = rec_space.element(image_N0[time_itvs // 4 * 2])
    rec_result_3 = rec_space.element(image_N0[time_itvs // 4 * 3])
    rec_result = rec_space.element(image_N0[time_itvs])
    ##%%
    # Plot the results of interest
    plt.figure(2, figsize=(24, 24))
    #plt.clf()

    plt.subplot(3, 3, 1)
    plt.imshow(np.rot90(template), cmap='bone',
               vmin=mini,
               vmax=maxi)
    plt.axis('off')
    #plt.savefig("/home/chchen/SwedenWork_Chong/NumericalResults_S/LDDMM_results/J_V/template_J.png", bbox_inches='tight')
    plt.colorbar()
    #plt.title('Trajectory from EllipseMvt with DeformationModulesODL/deform/vect_field_ellipse')

    plt.subplot(3, 3, 2)
    plt.imshow(np.rot90(rec_result_1), cmap='bone',
               vmin=mini,
               vmax=maxi)

    plt.axis('off')
    plt.colorbar()
    plt.title('time_pts = {!r}'.format(time_itvs // 4))

    plt.subplot(3, 3, 3)
    plt.imshow(np.rot90(rec_result_2), cmap='bone',
               vmin=mini,
               vmax=maxi)
    #grid=grid_points[time_itvs // 4].reshape(2, rec_space.shape[0], rec_space.shape[1]).copy()
    #plot_grid(grid, 2)
    plt.axis('off')
    plt.colorbar()
    plt.title('time_pts = {!r}'.format(time_itvs // 4 * 2))

    plt.subplot(3, 3, 4)
    plt.imshow(np.rot90(rec_result_3), cmap='bone',
               vmin=mini,
               vmax=maxi)
    #grid=grid_points[time_itvs // 4*2].reshape(2, rec_space.shape[0], rec_space.shape[1]).copy()
    #plot_grid(grid, 2)
    plt.axis('off')
    plt.colorbar()
    plt.title('time_pts = {!r}'.format(time_itvs // 4 * 3))

    plt.subplot(3, 3, 5)
    plt.imshow(np.rot90(rec_result), cmap='bone',
               vmin=mini,
               vmax=maxi)

    plt.axis('off')
    plt.colorbar()
    plt.title('time_pts = {!r}'.format(time_itvs ))

    plt.subplot(3, 3, 6)
    plt.imshow(np.rot90(target), cmap='bone',
               vmin=mini,
               vmax=maxi)
    plt.axis('off')
    plt.colorbar()
    plt.title('Ground truth')



    plt.savefig(name, bbox_inches='tight')
#

#%%
image_N0=odl.deform.ShootTemplateFromVectorFields(vector_fields_list, template)

name = '/home/bgris/Results/DeformationModules/testEstimation/Rotation/'
name += name_exp
#EstimatedTrajectory_SheppLogan_MultipleModule_LDDMMsigma_3_'
name += '_notnoisydirect'
#name += '_noisydirect'
#name += '_noisydirect_10angles'
plot_result(name, image_N0)

#%% Save vector fields
for i in range(nb_time_point_int + 1):
    np.savetxt(name + '_vectorfieldlist_t_' + str(i), vector_fields_list[i])
#
#%% Load vector fields
vector_fields_list_load = []
for i in range(nb_time_point_int + 1):
    vector_fields_list_load.append(np.loadtxt(name + '_vectorfieldlist_t_' + str(i)))
#


#%%
rec_result_1 = rec_space.element(image_N0[time_itvs // 4])
rec_result_2 = rec_space.element(image_N0[time_itvs // 4 * 2])
rec_result_3 = rec_space.element(image_N0[time_itvs // 4 * 3])
rec_result = rec_space.element(image_N0[time_itvs])

# Compute the projections of the reconstructed image
rec_proj_data = forward_op(rec_result)
#%%

for t in range(nb_time_point_int+1):
    #grid=grid_points[t].reshape(2, rec_space.shape[0], rec_space.shape[1]).copy()
    image_N0[t].show('t= {}'.format(t))
    #plot_grid(grid, 2)
#
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
#grid=grid_points[time_itvs // 4].reshape(2, rec_space.shape[0], rec_space.shape[1]).copy()
#plot_grid(grid, 2)
plt.axis('off')
plt.colorbar()
plt.title('time_pts = {!r}'.format(time_itvs // 4 * 2))

plt.subplot(3, 3, 4)
plt.imshow(np.rot90(rec_result_3), cmap='bone',
           vmin=np.asarray(rec_result_3).min(),
           vmax=np.asarray(rec_result_3).max())
#grid=grid_points[time_itvs // 4*2].reshape(2, rec_space.shape[0], rec_space.shape[1]).copy()
#plot_grid(grid, 2)
plt.axis('off')
plt.colorbar()
plt.title('time_pts = {!r}'.format(time_itvs // 4 * 3))

plt.subplot(3, 3, 5)
plt.imshow(np.rot90(rec_result), cmap='bone',
           vmin=np.asarray(rec_result).min(),
           vmax=np.asarray(rec_result).max())
#grid=grid_points[time_itvs // 4*3].reshape(2, rec_space.shape[0], rec_space.shape[1]).copy()
#plot_grid(grid, 2)
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

plt.subplot(3, 3, 7)
plt.plot(np.asarray(proj_data)[0], 'b', linewidth=1.0)
plt.plot(np.asarray(noise_proj_data)[0], 'r', linewidth=0.5)
plt.axis([0, int(round(rec_space.shape[0]*np.sqrt(2))), -5, 25]), plt.grid(True, linestyle='--')
#    plt.title('$\Theta=0^\circ$, b: truth, r: noisy, '
#        'g: rec_proj, SNR = {:.3}dB'.format(snr))
#    plt.gca().axes.yaxis.set_ticklabels([])

plt.subplot(3, 3, 8)
plt.plot(np.asarray(proj_data)[2], 'b', linewidth=1.0)
plt.plot(np.asarray(noise_proj_data)[2], 'r', linewidth=0.5)
plt.axis([0, int(round(rec_space.shape[0]*np.sqrt(2))), -5, 25]), plt.grid(True, linestyle='--')
#    plt.title('$\Theta=90^\circ$')
#    plt.gca().axes.yaxis.set_ticklabels([])

plt.subplot(3, 3, 9)
plt.plot(np.asarray(proj_data)[4], 'b', linewidth=1.0)
plt.plot(np.asarray(noise_proj_data)[4], 'r', linewidth=0.5)
plt.axis([int(round(rec_space.shape[0]*np.sqrt(2))), -5, 25]), plt.grid(True, linestyle='--')
#    plt.title('$\Theta=162^\circ$')
#    plt.gca().axes.yaxis.set_ticklabels([])'
#
plt.figure(2, figsize=(8, 1.5))
#plt.clf()
plt.plot(E)
plt.ylabel('Energy')
# plt.gca().axes.yaxis.set_ticklabels(['0']+['']*8)
plt.gca().axes.yaxis.set_ticklabels([])
plt.grid(True, linestyle='--')


#%% Save figures article
import odl
import numpy as np
from matplotlib import pylab as plt

from odl.discr import (uniform_discr, ResizingOperator)
typefig='pdf'
name_init_result = '/home/barbara/Results/LDDMM/SheppLogan/Rotation/Indirect/Rotationangles_100_LDDMM_sigma_2_t_'
name_init_figure = '/home/barbara/Dropbox/Recherche/mes_publi/ReconstructionDefMod/figures/SheppLogan_LDDMM_'

space_init = odl.uniform_discr(
    min_pt=[-16, -16], max_pt=[16, 16], shape=[512, 512],
    dtype='float32', interp='linear')

template_init=space_init.element(np.loadtxt(name_init_result + str(0)))
mini=0
maxi=1

n_iter = 20
for t in range(n_iter+1):
    im = space_init.element(np.loadtxt(name_init_result + str(t)))
    fig = im.show(clim=[mini,maxi])
    plt.axis('equal')
    plt.axis('off')
    fig.delaxes(fig.axes[1])
    plt.savefig(name_init_figure + str(t) + '.' +typefig, transparent = True, bbox_inches='tight',
        pad_inches = 0, format='pdf')
    plt.close('all')
#
    

# Target
if False:
    Name_target = '/home/barbara/data/SheppLogan/Rotation'
    im = space_init.element(np.loadtxt(Name_target))
    fig = im.show(clim=[mini,maxi])
    plt.axis('equal')
    plt.axis('off')
    fig.delaxes(fig.axes[1])
    plt.savefig(name_init_figure + 'ground_truth.'  + typefig, transparent = True, bbox_inches='tight',
        pad_inches = 0, format='pdf')
    plt.close('all')

# data
if False:
    ##%% Give the number of directions
    num_angles = 100
    min_angle = 0.0 * np.pi
    max_angle = 1. * np.pi
    
    # Create the uniformly distributed directions
    angle_partition = odl.uniform_partition(min_angle, max_angle, num_angles,
                                        nodes_on_bdry=[(True, True)])
    
    # Create 2-D projection domain
    # The length should be 1.5 times of that of the reconstruction space
    detector_partition = odl.uniform_partition(-20, 20, int(round(space_init.shape[0]*np.sqrt(2))))
    
    # Create 2-D parallel projection geometry
    geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)
    ## Ray transform aka forward projection. We use ASTRA CUDA backend.
    #forward_op = odl.tomo.RayTransform(space, geometry, impl='astra_cpu') 
    # Ray transform aka forward projection. We use skimage backend.
    forward_op = odl.tomo.RayTransform(space_init, geometry, impl='skimage')
    Name_target = '/home/barbara/data/SheppLogan/Rotation'
    target = space_init.element(np.loadtxt(Name_target))
    im = forward_op(target)
    fig = im.show()
    #plt.axis('equal')
    #plt.axis('off')
    fig.delaxes(fig.axes[1])
    plt.savefig(name_init_figure + 'data.'  + typefig, transparent = True, bbox_inches='tight',
        pad_inches = 0, format='pdf')
    plt.close('all')






