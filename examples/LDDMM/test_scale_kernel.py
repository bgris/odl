#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 16:29:26 2017

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



#
#%%

#path='/home/barbara'
path='/home/bgris'
I1name = path + '/odl/examples/LDDMM/v.png'
I0name = path + '/odl/examples/LDDMM/j.png'
#I1name = '/home/bgris/Downloads/pictures/v.png'
#I0name = '/home/bgris/Downloads/pictures/j.png'

I0 = np.rot90(plt.imread(I0name).astype('float'), -1)
I1 = np.rot90(plt.imread(I1name).astype('float'), -1)

# Discrete reconstruction space: discretized functions on the rectangle
space = odl.uniform_discr(
    min_pt=[-16, -16], max_pt=[16, 16], shape=I0.shape,
    dtype='float32', interp='linear')

# Create the ground truth as the given image
ground_truth = space.element(I1)



# Create the template as the given image
template = space.element(I0)
nb_sigma=500
sigma_list=np.linspace(space.cell_sides[0], 30,nb_sigma)

vector_fields_final_list=[]
attachment_term_final_list=[]
regularity_term_final_list=[]



# Maximum iteration number
niter = 500

# Give step size for solver
eps = 0.02

# Give regularization parameter
lamb = 1e-10

forward_op=odl.IdentityOperator(space)

# Create projection data by calling the op on the phantom
data = forward_op(ground_truth)
# Add white Gaussion noise onto the noiseless data
noise =0.005 * odl.phantom.noise.white_noise(forward_op.range)

# Create the noisy projection data
noise_proj_data = data + noise
data=[noise_proj_data.copy()]
data_time_points=np.array([1])
forward_operators=[forward_op]
Norm=odl.solvers.L2NormSquared(forward_op.range)
# Give the number of time points
time_itvs = 10
nb_time_point_int=time_itvs

for i in range(nb_sigma):
    eps = 0.02
    sigma=sigma_list[i]
    def kernel(x):
        scaled = [xi ** 2 / (2 * sigma ** 2) for xi in x]
        return np.exp(-sum(scaled))

    energy_op=odl.deform.TemporalAttachmentLDDMMGeom(nb_time_point_int, template ,data,
                            data_time_points, forward_operators,Norm, kernel,
                            domain=None)

    Reg=odl.deform.RegularityLDDMM(kernel,energy_op.domain)
    functional = energy_op + lamb*Reg

    vector_fields_list_init=energy_op.domain.zero()
    vector_fields_list=vector_fields_list_init.copy()
    attachment_term=energy_op(vector_fields_list)
    print(" Initial sigma i= {},  attachment term : {}".format(i,attachment_term))

    for k in range(niter):
        grad=functional.gradient(vector_fields_list)
        vector_fields_list_temp= (vector_fields_list- eps *grad).copy()
        attachment_term_temp=energy_op(vector_fields_list_temp)
        if(attachment_term_temp<attachment_term):
            vector_fields_list=vector_fields_list_temp.copy()
            attachment_term=attachment_term_temp
            eps*=1.2
            cont=0
            print("sigma i= {}, iter : {}  ,  attachment term : {}".format(i,k,attachment_term))
        else:
            eps*=0.8
            cont+=1
            print("sigma i= {}, iter : {}  ,  eps : {}".format(i,k,eps))
        if cont==20:
            break

    print(" final sigma i= {}, iter : {}  ,  attachment term : {}".format(i,k,attachment_term))

    name= path + '/Results/LDDMM/testscale/vectfieldopt_sigma_i_{}_'.format(i)

    for t in range(nb_time_point_int):
        np.savetxt(name + '{}'.format(t),vector_fields_list[t])

    attachment_term_final_list.append(attachment_term)
#


#%% Plot

plt.plot(sigma_list,attachment_term_final_list)



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
grid_points=compute_grid_deformation_list_bis(vector_fields_list, 1/nb_time_point_int, template.space.points().T)

#
#for t in range(nb_time_point_int):
#    grid=grid_points[t].reshape(2, 128, 128).copy()
#plot_grid(grid, 2)

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
