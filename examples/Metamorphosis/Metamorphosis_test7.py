#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 16:04:27 2018

@author: bgris
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 17:57:08 2017

@author: bgris
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 16:44:11 2017

@author: bgris
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 15:37:44 2017

@author: bgris
"""

import odl
import numpy as np
from matplotlib import pylab as plt
import os
##%%
#namepath= 'barbara'
namepath= 'bgris'

## Data parameters
index_name_template = 0
index_name_ground_truth = 0

index_angle = 0
index_maxangle = 0
index_noise = 2

## The parameter for kernel function
sigma = 300
name_sigma=str(int(sigma))

niter=300
epsV=0.02
epsZ=0.002
## Give regularization parameter
lamb = 1e-5
name_lamb='1e_' + str(-int(np.log(lamb)/np.log(10)))
tau = 1e-5
name_tau='1e_' + str(-int(np.log(tau)/np.log(10)))

# Give the number of time points
time_itvs = 20
nb_time_point_int=time_itvs




name_list_template = ['SheppLogan0', 'SheppLogan5']
name_list_ground_truth = ['SheppLogan5_deformed']
num_angles_list = [10, 50, 100]


maxiangle_list = ['pi', '0_25pi']
max_angle_list = [np.pi, 0.25*np.pi]
noise_level_list = [0.0, 0.05, 0.25]
noi_list = ['0', '0_05', '0_25']

name_val_template = name_list_template[index_name_template]
name_val = name_list_ground_truth[index_name_ground_truth]
num_angles = num_angles_list[index_angle]
maxiangle = maxiangle_list[index_maxangle]
max_angle = max_angle_list[index_maxangle]
noise_level = noise_level_list[index_noise]
noi = noi_list[index_noise]
min_angle = 0.0

name_exp = name_val + 'num_angles_' + str(num_angles) + '_min_angle_0_max_angle_'
name_exp += maxiangle + '_noise_' + noi


path_data = '/home/' + namepath + '/Dropbox/Recherche/mes_publi/Metamorphosis_PDE_ODE/data/test7/'
path_result_init = '/home/' + namepath + '/Results/Metamorphosis/test7/'
#path_result_init = '/home/bgris/Dropbox/Recherche/mes_publi/Metamorphosis_PDE_ODE/Results/test2/'
path_result = path_result_init + name_exp + '__sigma_' + name_sigma + '__lamb_'
path_result += name_lamb + '__tau_' + name_tau + '__niter_' + str(niter) + '__ntimepoints_' + str(time_itvs) + '/'



path_result_init_dropbox = '/home/' + namepath + '/Dropbox/Recherche/mes_publi/Metamorphosis_PDE_ODE/Results_ODE/test7/'
path_result_dropbox = path_result_init_dropbox + name_exp + '__sigma_' + name_sigma + '__lamb_'
path_result_dropbox += name_lamb + '__tau_' + name_tau + '__niter_' + str(niter) + '__ntimepoints_' + str(time_itvs) + '/'



# Discrete reconstruction space: discretized functions on the rectangle
rec_space = odl.uniform_discr(
    min_pt=[-16, -16], max_pt=[16, 16], shape=[256, 256],
    dtype='float32', interp='linear')


name_ground_truth = path_data + name_val
ground_truth = rec_space.element(np.loadtxt(name_ground_truth))

name_template = path_data + name_val_template
template = rec_space.element(np.loadtxt(name_template))



# Give kernel function
def kernel(x):
    scaled = [xi ** 2 / (2 * sigma ** 2) for xi in x]
    return np.exp(-sum(scaled))



## Create forward operator
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


## load data

data_load = forward_op.range.element(np.loadtxt(path_data + name_exp))



data=[data_load]
#data=[proj_data]
data_time_points=np.array([1])
forward_operators=[forward_op]
Norm=odl.solvers.L2NormSquared(forward_op.range)


##%% Define energy operator

functional=odl.deform.TemporalAttachmentMetamorphosisGeom(nb_time_point_int,
                            lamb,tau,template ,data,
                            data_time_points, forward_operators,Norm, kernel,
                            domain=None)


##%%
##%% Gradient descent

X_init=functional.domain.zero()
X=X_init.copy()
energy=functional(X)

print(" Initial ,  energy : {}".format(energy))


for k in range(niter):
    #X[1][0].show("Iter = {}".format(k),clim=[-1,1])
    #grad=functional.gradient(X)
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
            epsV *= 1.1
            epsZ *= 1.1
            print(" iter : {}  , energy : {}, epsV = {} , epsZ = {}".format(k,energy,epsV, epsZ))
        else:
            print("epsV = {} , epsZ = {}".format(epsV, epsZ))



#

##%%

##%% Compute estimated trajectory
image_list_data=functional.ComputeMetamorphosis(X[0],X[1])


image_list=functional.ComputeMetamorphosisListInt(X[0],X[1])

template_evo=odl.deform.ShootTemplateFromVectorFields(X[0], template)

zeta_transp=odl.deform.ShootSourceTermBackwardlist(X[0],X[1])

image_evol=odl.deform.IntegrateTemplateEvol(template,zeta_transp,0,nb_time_point_int)
##%%
mini=-1
maxi=1
#


#
##%% save results

os.mkdir(path_result)
os.mkdir(path_result_dropbox)


for i in range(nb_time_point_int + 1):
    np.savetxt(path_result + 'Metamorphosis_t_' + str(i), image_list[i])
    np.savetxt(path_result + 'Image_t_' + str(i), image_evol[i])
    np.savetxt(path_result + 'Template_t_' + str(i), template_evo[i])
#

## save plot results


##%% Plot metamorphosis
image_N0_list= [image_list, template_evo, image_evol]
name_plot_list = ['metamorphosis', 'template', 'image']
proj_template = forward_op(template)

for index, image_N0, name_plot in zip(range(3), image_N0_list, name_plot_list):
    rec_result_1 = rec_space.element(image_N0[time_itvs // 4])
    rec_result_2 = rec_space.element(image_N0[time_itvs // 4 * 2])
    rec_result_3 = rec_space.element(image_N0[time_itvs // 4 * 3])
    rec_result = rec_space.element(image_N0[time_itvs])
    rec_proj_data = forward_op(rec_result)
    plt.figure(index, figsize=(24, 24))
    plt.subplot(3, 3, 1)
    plt.imshow(np.rot90(template), cmap='bone',
               vmin=mini,
               vmax=maxi)
    plt.axis('off')
    #plt.savefig("/home/chchen/SwedenWork_Chong/NumericalResults_S/LDDMM_results/J_V/template_J.png", bbox_inches='tight')
    plt.colorbar()
    plt.title(name_plot)

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
    plt.axis('off')
    plt.colorbar()
    plt.title('time_pts = {!r}'.format(time_itvs // 4 * 2))

    plt.subplot(3, 3, 4)
    plt.imshow(np.rot90(rec_result_3), cmap='bone',
               vmin=mini,
               vmax=maxi)
    plt.axis('off')
    plt.colorbar()
    plt.title('time_pts = {!r}'.format(time_itvs // 4 * 3))

    plt.subplot(3, 3, 5)
    plt.imshow(np.rot90(rec_result), cmap='bone',
               vmin=mini,
               vmax=maxi)
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
    ##%%
    plt.subplot(3, 3, 7)
    plt.plot(np.asarray(data_load)[0], 'b', linewidth=0.5)
    plt.plot(np.asarray(rec_proj_data)[0], 'r', linewidth=0.5)
    plt.plot(np.asarray(proj_template)[0], 'k', linewidth=0.5)
    plt.axis([0, int(round(rec_space.shape[0]*np.sqrt(2))), -4, 20]), plt.grid(True, linestyle='--')
    #    plt.title('$\Theta=0^\circ$, b: truth, r: noisy, '
    #        'g: rec_proj, SNR = {:.3}dB'.format(snr))
    #    plt.gca().axes.yaxis.set_ticklabels([])

    plt.subplot(3, 3, 8)
    plt.plot(np.asarray(data_load)[2], 'b', linewidth=0.5)
    plt.plot(np.asarray(rec_proj_data)[2], 'r', linewidth=0.5)
    plt.plot(np.asarray(proj_template)[2], 'k', linewidth=0.5)
    plt.axis([0, int(round(rec_space.shape[0]*np.sqrt(2))), -4, 20]), plt.grid(True, linestyle='--')

    plt.subplot(3, 3, 9)
    plt.plot(np.asarray(data_load)[4], 'b', linewidth=0.5)
    plt.plot(np.asarray(rec_proj_data)[4], 'r', linewidth=0.5)
    plt.plot(np.asarray(proj_template)[4], 'k', linewidth=0.5)
    plt.axis([0,int(round(rec_space.shape[0]*np.sqrt(2))), -5, 25]), plt.grid(True, linestyle='--')


    name=path_result_dropbox + name_plot + '.png'
    plt.savefig(name, bbox_inches='tight')
#

