#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 12:29:18 2017

@author: barbara
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


import scipy

#%%

#I1name = '/home/bgris/Downloads/pictures/v.png'
#I0name = '/home/bgris/Downloads/pictures/j.png'
#I0 = np.rot90(plt.imread(I0name).astype('float'), -1)
#I1 = np.rot90(plt.imread(I1name).astype('float'), -1)


space = odl.uniform_discr(
    min_pt=[-16, -16], max_pt=[16, 16], shape=[256,256],
    dtype='float32', interp='linear')

path='/home/bgris/Downloads/'


template0=space.element(np.loadtxt(path + 'SheppLoganModifiedSource'))
template= space.element(scipy.ndimage.filters.gaussian_filter(template0.asarray(),1))
ground_truth0=space.element(np.loadtxt(path + 'SheppLoganModifiedTarget'))
ground_truth= space.element(scipy.ndimage.filters.gaussian_filter(ground_truth0.asarray(),1))

template.show(clim=[-1,1])
ground_truth.show(clim=[-1,1])

# The parameter for kernel function
sigma = 5

# Give kernel function
def kernel(x):
    scaled = [xi ** 2 / (2 * sigma ** 2) for xi in x]
    return np.exp(-sum(scaled))

# Maximum iteration number
niter = 200

# Give step size for solver
eps = 0.02

# Give regularization parameter
lamb = 1*1e-15
tau = 1* 1e-2

# Give the number of directions
num_angles = 10

# Create the uniformly distributed directions
angle_partition = odl.uniform_partition(0.0, np.pi, num_angles,
                                    nodes_on_bdry=[(True, True)])

# Create 2-D projection domain
# The length should be 1.5 times of that of the reconstruction space
rec_space=space
detector_partition = odl.uniform_partition(-24, 24, int(round(rec_space.shape[0]*np.sqrt(2))))

# Create 2-D parallel projection geometry
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

# Ray transform aka forward projection. We use ASTRA CUDA backend.
forward_op = odl.tomo.RayTransform(rec_space, geometry, impl='astra_cpu')
#forward_op = odl.IdentityOperator(rec_space)

# Create projection data by calling the op on the phantom
proj_data = forward_op(ground_truth)

# Add white Gaussion noise onto the noiseless data
noise =0.25 * odl.phantom.noise.white_noise(forward_op.range)

# Create the noisy projection data
noise_proj_data = proj_data + noise


# Give the number of time points
time_itvs = 10
nb_time_point_int=time_itvs

data=[noise_proj_data]
data=[proj_data]
data_time_points=np.array([1])
forward_operators=[forward_op]
Norm=odl.solvers.L2NormSquared(forward_op.range)


#%% Define energy operator

functional=odl.deform.TemporalAttachmentMetamorphosisGeom(nb_time_point_int,
                            lamb,tau,template ,data,
                            data_time_points, forward_operators,Norm, kernel,
                            domain=None)


#%% Gradient descent
niter=200
epsV=0.002
epsZ=0.00002
X_init=functional.domain.zero()
X=X_init.copy()
energy=functional(X)
print(" Initial ,  energy : {}".format(energy))

for k in range(niter):
    #X[1][0].show("Iter = {}".format(k),clim=[-1,1])
    grad=functional.gradient(X)
    X_temp0=X.copy()
    X_temp0[0]= (X[0]- epsV *grad[0]).copy()
    X_temp0[1]= (X[1]- epsZ *grad[1]).copy()
    energy_temp0=functional(X_temp0)
    if energy_temp0<energy:
        X=X_temp0.copy()
        energy=energy_temp0
        epsV*=1.1
        epsZ*=1.1
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
            print(" iter : {}  , energy : {}, epsV = {} , epsZ = {}".format(k,energy,epsV, epsZ))
        else:
            print("epsV = {} , epsZ = {}".format(epsV, epsZ))



#

#%%
mini=-1
maxi=1

nameinit='/home/barbara/odl/examples/Metamorphosis/SheppLogan/Shepp_Logan_modifie_greyscale_modifed_no_noise'
name0= nameinit + 'Metamorphosis_nbangle_10_sigma_5_lam_1_e__15_tau_1_e__2_iter_200'
#name0= nameinit + 'LDDMM_nbangle_10_sigma_5_lam_1_e__15_iter_200'


##%% fbp
#lam_fbp=200
#fbp = odl.tomo.fbp_op(forward_op, filter_type='Hann', frequency_scaling=lam_fbp)
#reco_fbp=fbp(data[0])
##reco_fbp.show()
##reco_fbp.show(clim=[-0.2,1.2])
#
#plt.figure(1, figsize=(24, 24))
#
#plt.imshow(np.rot90(reco_fbp), cmap='bone',
#           vmin=mini,
#           vmax=maxi)
#plt.colorbar()
#namefbp=nameinit + 'fbp.png'
#plt.savefig(namefbp, bbox_inches='tight')
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
##%%%
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
##%%%
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
##%%%
# Plot the results of interest
plt.figure(4, figsize=(24, 24))
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
