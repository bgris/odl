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

#I1name = '/home/bgris/Downloads/pictures/v.png'
#I0name = '/home/bgris/Downloads/pictures/j.png'
#I0 = np.rot90(plt.imread(I0name).astype('float'), -1)
#I1 = np.rot90(plt.imread(I1name).astype('float'), -1)
#

I0name = '//home/barbara/Téléchargements/code_for_LDDMM---triangles/ss_save.png' # 438 * 438, I0[:,:,1]
I1name = '//home/barbara/Téléchargements/code_for_LDDMM---triangles/ss_save_1.png' # 438 * 438, I0[:,:,1]
I0 = np.rot90(plt.imread(I0name).astype('float'), -1)[:,:, 1]
I1 = np.rot90(plt.imread(I1name).astype('float'), -1)[:,:, 1]


# Discrete reconstruction space: discretized functions on the rectangle
rec_space = odl.uniform_discr(
    min_pt=[-16, -16], max_pt=[16, 16], shape=I0.shape,
    dtype='float32', interp='linear')

# Create the ground truth as the given image
ground_truth = rec_space.element(I1)


# Create the template as the given image
template = 0.1*rec_space.element(I0)


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

#ground_truth.show('ground truth')
#template.show('template')

# The parameter for kernel function
sigma = 2.0

# Give kernel function
def kernel(x):
    scaled = [xi ** 2 / (2 * sigma ** 2) for xi in x]
    return np.exp(-sum(scaled))

# Maximum iteration number
niter = 200

# Give step size for solver
eps = 0.02

# Give regularization parameter
lamb = 0.5*1e-11
tau = 0.5* 1e-2

# Give the number of directions
num_angles = 10

# Create the uniformly distributed directions
angle_partition = odl.uniform_partition(0.0, np.pi, num_angles,
                                    nodes_on_bdry=[(True, True)])

# Create 2-D projection domain
# The length should be 1.5 times of that of the reconstruction space
detector_partition = odl.uniform_partition(-24, 24, int(round(rec_space.shape[0]*np.sqrt(2))))

# Create 2-D parallel projection geometry
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

# Ray transform aka forward projection. We use ASTRA CUDA backend.
forward_op = odl.tomo.RayTransform(rec_space, geometry, impl='astra_cpu')
#forward_op = odl.IdentityOperator(rec_space)

# Create projection data by calling the op on the phantom
proj_data = forward_op(ground_truth)

# Add white Gaussion noise onto the noiseless data
noise =1.0 * odl.phantom.noise.white_noise(forward_op.range)

# Create the noisy projection data
noise_proj_data = proj_data + noise

# Compute the signal-to-noise ratio in dB
snr = snr_fun(proj_data, noise_proj_data - proj_data, impl='dB')

# Output the signal-to-noise ratio
print('snr = {!r}'.format(snr))

# Give the number of time points
time_itvs = 5
nb_time_point_int=time_itvs

data=[noise_proj_data]
#data=[proj_data]
data_time_points=np.array([1])
forward_operators=[forward_op]
Norm=odl.solvers.L2NormSquared(forward_op.range)


#%% Define energy operator

functional=odl.deform.TemporalAttachmentMetamorphosisGeom(nb_time_point_int,
                            lamb,tau,template ,data,
                            data_time_points, forward_operators,Norm, kernel,
                            domain=None)

nameinit='/home/barbara/odl/examples/Metamorphosis/Triangles/triangle_fac_greyscale_0_1_'
name0= nameinit + 'Metamorphosis_sigma_2_angle_10_lam_0_5_e__11_tau_0_5_e__2_iter_100'    
#%% Gradient descent
niter=100
epsV=0.02
epsZ=0.002
X_init=functional.domain.zero()
X=X_init.copy()
energy=functional(X)
print(" Initial ,  energy : {}".format(energy))

for k in range(niter):
    grad=functional.gradient(X)
    X[0]= (X[0]- epsV *grad[0]).copy()
    X[1]= (X[1]- epsZ *grad[1]).copy()
    energy=functional(X)
    print(" iter : {}  , energy : {}".format(k,energy))

#

#%%
lam_fbp=0.5
fbp = odl.tomo.fbp_op(forward_op, filter_type='Hann', frequency_scaling=lam_fbp)
reco_fbp=fbp(data[0])
#reco_fbp.show()
#reco_fbp.show(clim=[-0.2,1.2])

plt.figure(5, figsize=(24, 24))
mini=0
maxi=1
plt.imshow(np.rot90(reco_fbp), cmap='bone',
           vmin=mini,
           vmax=maxi)
plt.colorbar()
namefbp=nameinit + 'fbp.png'
plt.savefig(namefbp, bbox_inches='tight')
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
#    image_evol[i].show('Image evolution time {} '.format(i),clim=[-0.2,1.2])
#
mini=0
maxi=1
#
#grid_points=compute_grid_deformation_list(X[0], 1/nb_time_point_int, template.space.points().T)

#
#for t in range(nb_time_point_int):
#    grid=grid_points[t].reshape(2, 128, 128).copy()
#plot_grid(grid, 2)
#%%
#image_list=odl.ProductSpace(functional.template.space,functional.N).element()
#zeta_transp=odl.deform.ShootSourceTermBackwardlist(X[0], X[1]).copy()
#template_evolution=odl.deform.IntegrateTemplateEvol(functional.template,zeta_transp,0,functional.N)
#odl.deform.ShootTemplateFromVectorFieldsFinal(X[0],template_evolution[k],0,k).copy()

#%% Plot metamorphosis
image_N0= image_list
rec_result_1 = rec_space.element(image_N0[time_itvs // 4])
rec_result_2 = rec_space.element(image_N0[time_itvs // 4 * 2])
rec_result_3 = rec_space.element(image_N0[time_itvs // 4 * 3])
rec_result = rec_space.element(image_N0[time_itvs])

# Compute the projections of the reconstructed image
rec_proj_data = forward_op(rec_result)
##%%
#
#for t in range(nb_time_point_int+1):
#    #grid=grid_points[t].reshape(2, rec_space.shape[0], rec_space.shape[1]).copy()
#    image_N0[t].show('t= {}'.format(t))
#    #plot_grid(grid, 2)
##
#%%%
# Plot the results of interest
plt.figure(1, figsize=(24, 24))
#plt.clf()
mini=0
maxi=1
plt.subplot(3, 3, 1)
plt.imshow(np.rot90(template), cmap='bone',
           vmin=mini,
           vmax=maxi)
plt.axis('off')
#plt.savefig("/home/chchen/SwedenWork_Chong/NumericalResults_S/LDDMM_results/J_V/template_J.png", bbox_inches='tight')
plt.colorbar()
plt.title('Metamorphosis')

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
<<<<<<< d595d3e68921fe9294c4f8a3716855c641ae99f2
#grid=grid_points[time_itvs // 4*2].reshape(2, rec_space.shape[0], rec_space.shape[1]).copy()
#plot_grid(grid, 2)
=======
>>>>>>> examples
plt.axis('off')
plt.colorbar()
plt.title('time_pts = {!r}'.format(time_itvs // 4 * 3))

plt.subplot(3, 3, 5)
plt.imshow(np.rot90(rec_result), cmap='bone',
           vmin=mini,
           vmax=maxi)
#grid=grid_points[time_itvs // 4*3].reshape(2, rec_space.shape[0], rec_space.shape[1]).copy()
#plot_grid(grid, 2)
plt.axis('off')
plt.colorbar()
plt.title('Reconstructed by {!r} iters, '
    '{!r} projs'.format(niter, num_angles))

plt.subplot(3, 3, 6)
plt.imshow(np.rot90(ground_truth), cmap='bone',
           vmin=mini,
           vmax=maxi)
plt.axis('off')
plt.colorbar()
plt.title('Ground truth')
#%%
plt.subplot(3, 3, 7)
plt.plot(np.asarray(proj_data)[0], 'b', linewidth=1.0)
plt.plot(np.asarray(noise_proj_data)[0], 'r', linewidth=0.5)
plt.axis([0, int(round(rec_space.shape[0]*np.sqrt(2))), -4, 20]), plt.grid(True, linestyle='--')
#    plt.title('$\Theta=0^\circ$, b: truth, r: noisy, '
#        'g: rec_proj, SNR = {:.3}dB'.format(snr))
#    plt.gca().axes.yaxis.set_ticklabels([])

plt.subplot(3, 3, 8)
plt.plot(np.asarray(proj_data)[2], 'b', linewidth=1.0)
plt.plot(np.asarray(noise_proj_data)[2], 'r', linewidth=0.5)
plt.axis([0, int(round(rec_space.shape[0]*np.sqrt(2))), -4, 20]), plt.grid(True, linestyle='--')
#    plt.title('$\Theta=90^\circ$')
#    plt.gca().axes.yaxis.set_ticklabels([])

plt.subplot(3, 3, 9)
plt.plot(np.asarray(proj_data)[4], 'b', linewidth=1.0)
plt.plot(np.asarray(noise_proj_data)[4], 'r', linewidth=0.5)
plt.axis([0,int(round(rec_space.shape[0]*np.sqrt(2))), -5, 25]), plt.grid(True, linestyle='--')
#    plt.title('$\Theta=162^\circ$')
#    plt.gca().axes.yaxis.set_ticklabels([])'
##
#plt.figure(2, figsize=(8, 1.5))
##plt.clf()
#plt.plot(E)
#plt.ylabel('Energy')
## plt.gca().axes.yaxis.set_ticklabels(['0']+['']*8)
#plt.gca().axes.yaxis.set_ticklabels([])
#plt.grid(True, linestyle='--')

name=name0 + 'metamorphosis.png'
plt.savefig(name, bbox_inches='tight')

#%% Plot template
image_N0= template_evo
rec_result_1 = rec_space.element(image_N0[time_itvs // 4])
rec_result_2 = rec_space.element(image_N0[time_itvs // 4 * 2])
rec_result_3 = rec_space.element(image_N0[time_itvs // 4 * 3])
rec_result = rec_space.element(image_N0[time_itvs])
##%%
#
#for t in range(nb_time_point_int+1):
#    #grid=grid_points[t].reshape(2, rec_space.shape[0], rec_space.shape[1]).copy()
#    image_N0[t].show('t= {}'.format(t))
#    #plot_grid(grid, 2)
##
#%%%
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
plt.title('Template')

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
#grid=grid_points[time_itvs // 4*3].reshape(2, rec_space.shape[0], rec_space.shape[1]).copy()
#plot_grid(grid, 2)
plt.axis('off')
plt.colorbar()
plt.title('Reconstructed by {!r} iters, '
    '{!r} projs'.format(niter, num_angles))

plt.subplot(3, 3, 6)
plt.imshow(np.rot90(ground_truth), cmap='bone',
           vmin=mini,
           vmax=maxi)
plt.axis('off')
plt.colorbar()
plt.title('Ground truth')


name=name0 + 'template.png'
plt.savefig(name, bbox_inches='tight')

#%% Plot metamorphosis
image_N0= image_evol
rec_result_1 = rec_space.element(image_N0[time_itvs // 4])
rec_result_2 = rec_space.element(image_N0[time_itvs // 4 * 2])
rec_result_3 = rec_space.element(image_N0[time_itvs // 4 * 3])
rec_result = rec_space.element(image_N0[time_itvs])

##%%
#
#for t in range(nb_time_point_int+1):
#    #grid=grid_points[t].reshape(2, rec_space.shape[0], rec_space.shape[1]).copy()
#    image_N0[t].show('t= {}'.format(t))
#    #plot_grid(grid, 2)
##
#%%%
# Plot the results of interest
plt.figure(3, figsize=(24, 24))
#plt.clf()

plt.subplot(3, 3, 1)
plt.imshow(np.rot90(template), cmap='bone',
           vmin=mini,
           vmax=maxi)
plt.axis('off')
#plt.savefig("/home/chchen/SwedenWork_Chong/NumericalResults_S/LDDMM_results/J_V/template_J.png", bbox_inches='tight')
plt.colorbar()
plt.title('Image')

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
#grid=grid_points[time_itvs // 4*3].reshape(2, rec_space.shape[0], rec_space.shape[1]).copy()
#plot_grid(grid, 2)
plt.axis('off')
plt.colorbar()
plt.title('Reconstructed by {!r} iters, '
    '{!r} projs'.format(niter, num_angles))

plt.subplot(3, 3, 6)
plt.imshow(np.rot90(ground_truth), cmap='bone',
           vmin=mini,
           vmax=maxi)
plt.axis('off')
plt.colorbar()
plt.title('Ground truth')

#    plt.title('$\Theta=90^\circ$')
#    plt.gca().axes.yaxis.set_ticklabels([])


name=name0 + 'image.png'
plt.savefig(name, bbox_inches='tight')

