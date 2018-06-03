#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 12:57:10 2017

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
import matplotlib.pyplot as plt

import scipy


# Discrete reconstruction space: discretized functions on the rectangle
space = odl.uniform_discr(
    min_pt=[-16, -16], max_pt=[16, 16], shape=[256,256],
    dtype='float32', interp='linear')

numtest = 1

## Data parameters
index_name_template = 1
index_name_ground_truth = 0

index_angle = 2
index_maxangle = 0
index_noise = 0

typefig = '.pdf'
## The parameter for kernel function
sigma = 2.0
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
time_itvs = 10
nb_time_point_int=time_itvs




name_list_template = ['template_values__0_2__0_9', 'template_values__0__1']
name_list_ground_truth = ['ground_truth_values__0_2__0_9', 'ground_truth_values__0__1']
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


path_data = '/home/bgris/Dropbox/Recherche/mes_publi/Metamorphosis_PDE_ODE/data/test' + str(numtest) + '/'
path_result_init = '/home/bgris/Results/Metamorphosis/test' + str(numtest) + '/LDDMM/'
path_result = path_result_init + name_exp + '__sigma_' + name_sigma + '__lamb_'
path_result += name_lamb + '__tau_' + name_tau + '__niter_' + str(niter) + '__ntimepoints_' + str(time_itvs) + '/'


path_figure = '/home/bgris/Dropbox/Recherche/presentations/CambridgeDec2017/figures/'
name_figure = path_figure + 'test' + str(numtest) + 'LDDMM' + name_exp + '__sigma_' + name_sigma + '__lamb_'
name_figure += name_lamb + '__tau_' + name_tau + '__niter_' + str(niter) + '__ntimepoints_' + str(time_itvs)


##%%
mini=-1
maxi=1

image = space.element(np.loadtxt(path_result + '_' + str(time_itvs)))
fig = image.show(clim=[mini, maxi])
plt.axis('off')
fig.delaxes(fig.axes[1])
plt.savefig(name_figure +  typefig, transparent = True, bbox_inches='tight',
    pad_inches = 0)
#


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


rec_proj_data = forward_op(space.element(np.loadtxt(path_result  + '_' + str(time_itvs))))
data_load = forward_op.range.element(np.loadtxt(path_data + name_exp))
proj_template = forward_op(space.element(np.loadtxt(path_data + name_list_template[index_name_template] )))

indexdataplot = 0
plt.figure()
plt.plot(np.asarray(data_load)[indexdataplot], 'b', linewidth=0.5, label = 'Data')
plt.plot(np.asarray(rec_proj_data)[indexdataplot], 'r', linewidth=0.5, label = 'Result')
plt.plot(np.asarray(proj_template)[indexdataplot], 'k', linewidth=0.5, label = 'Template data')
plt.axis([0, int(round(space.shape[0]*np.sqrt(2))), -4, 20]), plt.grid(True, linestyle='--')
plt.legend()
plt.savefig(name_figure + 'DataResult' + typefig, transparent = True, bbox_inches='tight',
    pad_inches = 0)




