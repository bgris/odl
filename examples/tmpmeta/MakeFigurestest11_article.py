#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 10:21:34 2018

@author: bgris
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 09:58:55 2018

@author: bgris
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 15:40:26 2018

@author: bgris
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 12:52:58 2017

@author: bgris
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 12:30:26 2017

@author: bgris
"""


import odl
import numpy as np

##%% Create data from lddmm registration
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display, Image
namepath = 'bgris'



# Discrete reconstruction space: discretized functions on the rectangle
space = odl.uniform_discr(
    min_pt=[-16, -16], max_pt=[16, 16], shape=[256,256],
    dtype='float32', interp='linear')

numtest = 11

## Data parameters
index_name_template = 0
index_name_ground_truth = 4

index_angle = 3
index_maxangle = 0
index_minangle = 0
index_noise = 2


## The parameter for kernel function
sigma = 3.0
name_sigma=str(int(sigma))

niter=1000
epsV=0.02
epsZ=0.002
## Give regularization parameter
lamb = 1e-6
name_lamb='1e_' + str(-int(np.log(lamb)/np.log(10)))
tau = 1e-4
name_tau='1e_' + str(-int(np.log(tau)/np.log(10)))

# Give the number of time points
time_itvs = 20
nb_time_point_int=time_itvs

typefig = '.pdf'


name_list_template = ['SheppLogan10']
name_list_ground_truth = ['SheppLogan9_deformed', 'SheppLogan10_deformed', 'SheppLogan11_deformed', 'SheppLogan12_deformed', 'SheppLogan13_deformed']
num_angles_list = [10, 50, 100, 20, 30]
maxiangle_list = ['pi', '0_25pi', '0_75pi']
max_angle_list = [np.pi, 0.25*np.pi, 0.75*np.pi]
noise_level_list = [0.0, 0.05, 0.25]
noi_list = ['0', '0_05', '0_25']
miniangle_list = ['0', '0_25pi']
min_angle_list = [0.0, 0.25*np.pi]

name_val_template = name_list_template[index_name_template]
name_val = name_list_ground_truth[index_name_ground_truth]
num_angles = num_angles_list[index_angle]
maxiangle = maxiangle_list[index_maxangle]
max_angle = max_angle_list[index_maxangle]
noise_level = noise_level_list[index_noise]
noi = noi_list[index_noise]
miniangle = miniangle_list[index_minangle]
min_angle = min_angle_list[index_minangle]

name_exp = name_val + 'num_angles_' + str(num_angles) + '_min_angle_' + miniangle + '_max_angle_'
name_exp += maxiangle + '_noise_' + noi


#path_data = '/home/bgris/data/test' + str(numtest) + '/'
path_data = '/home/bgris/data/Metamorphosis/test' + str(11) + '/'
path_result_init = '/home/bgris/Results/Metamorphosis/test' + str(numtest) + '/'
path_result = path_result_init + name_exp + '__sigma_' + name_sigma + '__lamb_'
path_result += name_lamb + '__tau_' + name_tau + '__niter_' + str(niter) + '__ntimepoints_' + str(time_itvs) + '/'


path_figure = '/home/' + namepath + '/Dropbox/Recherche/mes_publi/ReconstructionMetamorphosis/figures/'
name_figure = path_figure + 'test' + str(numtest) + name_exp + '__sigma_' + name_sigma + '__lamb_'
name_figure += name_lamb + '__tau_' + name_tau + '__niter_' + str(niter) + '__ntimepoints_'


#%%
mini=0
maxi=1

name_list = ['Metamorphosis', 'Image', 'Template']
for name in name_list:
#    for t in range(time_itvs + 1):
        t = time_itvs
        image = np.rot90(space.element(np.loadtxt(path_result + name + '_t_' + str(t))))
        plt.imshow(image, cmap=plt.get_cmap('bone'), vmin=mini, vmax=maxi)
        plt.axis('off')
#        fig.delaxes(fig.axes[1])
        plt.savefig(name_figure + name  + str(t) + typefig, transparent = True, bbox_inches='tight',
        pad_inches = 0)
#

plt.close('all')

#%%
## Create forward operator
## Create the uniformly distributed directions
angle_partition = odl.uniform_partition(min_angle, max_angle, num_angles,
                                    nodes_on_bdry=[(True, True)])

## Create 2-D projection domain
## The length should be 1.5 times of that of the reconstruction space
detector_partition = odl.uniform_partition(-24, 24, int(round(space.shape[0]*np.sqrt(2))))

## Create 2-D parallel projection geometry
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

## Ray transform aka forward projection. We use ASTRA CUDA backend.
forward_op = odl.tomo.RayTransform(space, geometry, impl='astra_cpu')

#rec_proj_data = forward_op(space.element(np.loadtxt(path_data + 'Metamorphosis' + '_t_' + str(time_itvs))))
data_load = forward_op.range.element(np.loadtxt(path_data + name_exp))
proj_template = forward_op(space.element(np.loadtxt(path_data + name_list_template[index_name_template] )))

indexdataplot = 10
plt.figure()
plt.plot(np.asarray(data_load)[indexdataplot], 'b', linewidth=0.5, label = 'Data')
#plt.plot(np.asarray(rec_proj_data)[indexdataplot], 'r', linewidth=0.5, label = 'Result')
plt.plot(np.asarray(proj_template)[indexdataplot], 'k', linewidth=0.5, label = 'Template data')
plt.axis([0, int(round(space.shape[0]*np.sqrt(2))), -4, 20]), plt.grid(True, linestyle='--')
plt.legend()
plt.savefig(name_figure + 'Data' + str(indexdataplot) + typefig, transparent = True, 
            bbox_inches= matplotlib.transforms.Bbox([[0,-0.5], [5.5, 5]]))


# figures for template
if False:
    image = np.rot90(space.element(np.loadtxt(path_data + name_list_template[index_name_template] )))
    plt.imshow(image, cmap=plt.get_cmap('bone'), vmin=mini, vmax=maxi)
    plt.savefig(path_figure + 'test' + str(numtest) + 'Template_withaxis' + typefig, transparent = True, bbox_inches='tight',
    pad_inches = 0)


# figures for ground truth
if False:
    numtruth = 0
    image = np.rot90(space.element(np.loadtxt(path_data + name_list_ground_truth[numtruth] )))
    plt.axis('off')
    plt.imshow(image, cmap=plt.get_cmap('bone'), vmin=mini, vmax=maxi)
    plt.savefig( path_figure + 'test' + str(numtest) + name_list_ground_truth[numtruth] + typefig, transparent = True, bbox_inches='tight',
    pad_inches = 0)

#%%

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

#
        

name_exp = name_val + 'num_angles_' + str(num_angles) + '_min_angle_0_max_angle_'
name_exp += maxiangle + '_noise_0'

data_no_noise = forward_op.range.element(np.loadtxt(path_data + name_exp))






print('SNR = {}'.format(snr_fun(data_no_noise, data_load - data_no_noise, 'dB')))       


#%%  Pour TV
##%%
mini=-1
maxi=1
#lam = 1
data_matching = 'exact'
data_matching_list = ['exact', 'inexact']
for data_matching in data_matching_list :
    image = np.rot90(space.element(np.loadtxt(path_result + '_TV_' + data_matching + 'num_angles_' + str(num_angles) + '__lam_' + str(lam))))
    plt.axis('off')
    plt.imshow(image, cmap=plt.get_cmap('bone'), vmin=mini, vmax=maxi)
    plt.savefig(path_figure + 'test' + str(numtest) + 'TV' + data_matching+ 'num_angles_' + str(num_angles)  + '__lam_' + str(lam) + typefig, transparent = True, bbox_inches='tight',
        pad_inches = 0)
#

#%%  Pour FBP
##%%
mini=-1
maxi=1
image = np.rot90(space.element(np.loadtxt(path_result + '_FBP_' + 'num_angles_' + str(num_angles))))
plt.imshow(image, cmap=plt.get_cmap('bone'), vmin=mini, vmax=maxi)
plt.axis('off')
plt.savefig(path_figure + 'test' + str(numtest) + '_FBP_' +'num_angles_' + str(num_angles) + typefig, transparent = True, bbox_inches='tight',
pad_inches = 0)


#%%



# Discrete reconstruction space: discretized functions on the rectangle
rec_space = odl.uniform_discr(
    min_pt=[-16, -16], max_pt=[16, 16], shape=[256, 256],
    dtype='float32', interp='linear')


name_ground_truth = path_data + name_val
ground_truth = rec_space.element(np.loadtxt(name_ground_truth))

name_template = path_data + name_val_template
template = rec_space.element(np.loadtxt(name_template))
image_list=[]
template_evo=[]
image_evol = []

for i in range(nb_time_point_int + 1):
    image_list.append(np.loadtxt(path_result + 'Metamorphosis_t_' + str(i)))
    template_evo.append( np.loadtxt(path_result + 'Image_t_' + str(i)))
    image_evol.append(np.loadtxt(path_result + 'Template_t_' + str(i)))
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


    name=path_result + name_plot + '.png'
    plt.savefig(name, bbox_inches='tight')
#
plt.close('all')