#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 18:42:38 2018

@author: barbara
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 16:57:36 2018

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
#namepath= 'gris'

## Data parameters
index_name_template = 0
index_name_ground_truth = 0
nb_data_point = 10
indexes_name_ground_truth_timepoints = [i + 1 for i in range(nb_data_point)]
data_time_points=np.array([ (i+1)/10 for i in range(nb_data_point)])

index_angle = 4
index_maxangle = 0
index_noise = 2



## The parameter for kernel function
sigma = 3.0
name_sigma=str(int(sigma))

niter=100
epsV=0.02
epsZ=0.02
## Give regularization parameter
lamb = 1e-5
name_lamb='1e_' + str(-int(np.log(lamb)/np.log(10)))
tau = 1e-5
name_tau='1e_' + str(-int(np.log(tau)/np.log(10)))

# Give the number of time points
time_itvs = 10
nb_time_point_int=time_itvs
numtest = 16

nb_data_points = len(indexes_name_ground_truth_timepoints)

#name_list_template = ['SheppLogan7_0']
#name_list_ground_truth = ['SheppLogan7_']
name_list_template = [ 'temporal__t_0']
name_list_ground_truth = [ 'temporal__t_']

num_angles_list = [10, 20, 30, 50, 100]
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
miniangle = '0'

name_exp = name_val + 'num_angles_' + str(num_angles) + '_min_angle_0_max_angle_' + maxiangle + '_noise_' + noi
#name_list = [name_val + 'num_angles_' + str(num_angles) + '_min_angle_0_max_angle_' + maxiangle + '_noise_' + noi for i in range(nb_data_points)]
name_list = [name_val + str(i) for i in range(nb_data_points + 1)]


path_data = '/home/' + namepath + '/data/Metamorphosis/test' + str(numtest) + '/'
path_result_init = '/home/' + namepath + '/Results/Metamorphosis/test' + str(numtest) + '/'
#path_result_init = '/home/bgris/Dropbox/Recherche/mes_publi/Metamorphosis_PDE_ODE/Results/test2/'
path_result = path_result_init + name_exp + '__sigma_' + name_sigma + '__lamb_'
path_result += name_lamb + '__tau_' + name_tau + '__niter_' + str(niter) + '__ntimepoints_' + str(time_itvs) + 'datatimepoints' + str(len(data_time_points)) + '/'



#path_result_init_dropbox = '/home/' + namepath + '/Dropbox/Recherche/mes_publi/Metamorphosis_PDE_ODE/Results_ODE/test6/'
#path_result_dropbox = path_result_init_dropbox + name_exp + '__sigma_' + name_sigma + '__lamb_'
#path_result_dropbox += name_lamb + '__tau_' + name_tau + '__niter_' + str(niter) + '__ntimepoints_' + str(time_itvs) + '/'



# Discrete reconstruction space: discretized functions on the rectangle
rec_space = odl.uniform_discr(
    min_pt=[-16, -16], max_pt=[16, 16], shape=[256, 256],
    dtype='float32', interp='linear')


name_ground_truth = [path_data + name_val + str(indexes_name_ground_truth_timepoints[i])  for i in range(nb_data_points)]
ground_truth_list = [rec_space.element(np.loadtxt(name_ground_truth[i])) for i in range(nb_data_points)]

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

#data_load = forward_op.range.element(np.loadtxt(path_data + name_exp))
nameexp =  'num_angles_' + str(num_angles) + '_min_angle_' + miniangle + '_max_angle_'
nameexp += maxiangle + '_noise_' + noi + 'randompartition'

name_data = [path_data + name_list[i+1] + nameexp for i in range(nb_data_points)]
data_load = [forward_op.range.element(np.loadtxt(name_data[i])) for i in range(nb_data_points)]


#data=[data_load]
#data=[proj_data]
forward_operators=[forward_op for i in range(nb_data_points)]
Norm=odl.solvers.L2NormSquared(forward_op.range)


#%% Define energy operator

functional=odl.deform.TemporalAttachmentMetamorphosisGeom(nb_time_point_int,
                            lamb,tau,template ,data_load,
                            data_time_points, forward_operators,Norm, kernel,
                            domain=None)




#%% load data
image_list = []
template_evo = []
image_evol = []
for i in range(nb_time_point_int + 1):
    image_list.append(rec_space.element(np.loadtxt(path_result + 'Metamorphosis_t_' + str(i))))
    template_evo.append(rec_space.element(np.loadtxt(path_result + 'Template_t_' + str(i))))
    image_evol.append(rec_space.element(np.loadtxt(path_result + 'Image_t_' + str(i))))
#

#%%
for i in range(nb_time_point_int + 1):
    image_list[i].show('Metamorphosis_t_' + str(i))
    template_evo[i].show('Template_t_' + str(i))
    image_evol[i].show( 'Image_t_' + str(i))
#

#%%
for i in range(nb_time_point_int ):
    (image_list[i+1] - ground_truth_list[i]).show('Difference_t_' + str(i))
#

#%% Plot in one big image
from mpl_toolkits.axes_grid1 import SubplotDivider, LocatableAxes, make_axes_locatable, Size
mini = -0.3
maxi = 1
image_N0_list= [ image_list, template_evo, image_evol]
names_list = ['Image', 'Deformation part', 'template part']
fig1 = plt.figure(figsize=(nb_time_point_int+1, 5*4))
#for i in range(nb_time_point_int + 1):
#    plt.text(-1, -20, 't = ' + str(i/nb_time_point_int), rotation = 90)
plt.subplot(nb_time_point_int + 1, 4,  1)
plt.title('Ground truth')
plt.axis('off')
i=0
for i in range(nb_time_point_int + 1):
    plt.text(-0.2, 0.5 - 1.2*i, 't = ' + str(i/nb_time_point_int), rotation = 90)
for i in range(nb_time_point_int + 1):
#    print(i)

    if (i < nb_time_point_int):
        plt.subplot(nb_time_point_int + 1, 4, (i+1)*4 + 1)
        plt.imshow(np.rot90(ground_truth_list[i]), cmap='bone',
               vmin=mini,
               vmax=maxi, aspect='auto')
        plt.axis('off')
#        plt.text(-3, 0.5, 't = ' + str((i+1)/nb_time_point_int), rotation = 90)

    for j in range(3):
        im=plt.subplot(nb_time_point_int + 1, 4, i*4 + j+ 2)
        if (i==0):
           plt.title(names_list[j]) 
           
        plt.imshow(np.rot90(image_N0_list[j][i]), cmap='bone',
               vmin=mini,
               vmax=maxi, aspect='auto')
        #if (i > 0 or j > 0):
        plt.axis('off')
        
    if i==0:
        plt.subplot(nb_time_point_int + 1, 4, 1)
        plt.colorbar()
#        im=plt.subplot(nb_time_point_int + 1, 4, 1)
#        divider = SubplotDivider(fig1, nb_time_point_int + 1, 4, 1, aspect=True)
#        ax_cb = LocatableAxes(fig1, divider.get_position())
#        ax = LocatableAxes(fig1, divider.get_position())
#        h = [Size.AxesX(ax),  # main axes
#             Size.Fixed(0.05),  # padding, 0.1 inch
#             Size.Fixed(0.2),  # colorbar, 0.3 inch
#             ]
#    
#        v = [Size.AxesY(ax)]
#    
#        divider.set_horizontal(h)
#        divider.set_vertical(v)
#    
#        ax_cb.set_axes_locator(divider.new_locator(nx=2, ny=0))
#    
#        fig1.add_axes(ax_cb)
#    
#        ax_cb.axis["left"].toggle(all=False)
#        ax_cb.axis["right"].toggle(ticks=True)
#        #ax = plt.gca()  
#        #divider = make_axes_locatable(ax)
#        #cax = divider.append_axes("right", size="5%", pad=0.05)
#        plt.colorbar(im, cax=ax_cb)
        #cax = plt.axes([0.85, 0.1, 0.075, 0.8])
        #if (i==0 and j==0):
        #    plt.colorbar(im)
name=path_result + 'plot.png'
plt.savefig(name, bbox_inches='tight')

#
#%% plot data
i=0
plt.figure(figsize=(1, 15))
plt.imshow(np.rot90(np.array(list_data[i])), cmap = 'bone')
plt.axis('off')
#fig = list_data[i].show()
plt.savefig(path_result + 'plotdata_t_' + str(i+1), bbox_inches='tight')


#%% Plot FBF and TV
names = ['_FBP_num_angles_10', '_TV_exactnum_angles_10__lam_1', '_TV_inexactnum_angles_10__lam_1']
i = 0
image = np.loadtxt(path_result + names[i])
plt.imshow(np.rot90(image), cmap = 'bone', vmin=mini, vmax=maxi)
plt.axis('off')
name=path_result + names[i] + 'plot.png'
plt.savefig(name, bbox_inches='tight')
#%% plot full data
mini = -2
maxi = 15
fig = data.show(clim=[mini, maxi])
name=path_result + '/plotfulldata.png'
#plt.axis([0, 100, 0, 20])
fig.savefig(name)

#%% plot data
k =9
fig = list_data[k].show(clim=[mini, maxi])
name=path_result + '/data' + str(k) + '.png'
#plt.axis([0, 100, 0, 20])
fig.savefig(name)

#%%
## save plot results

#
###%% Plot metamorphosis
#image_N0_list= [image_list, template_evo, image_evol]
#name_plot_list = ['metamorphosis', 'template', 'image']
#proj_template = forward_op(template)
#
#for index, image_N0, name_plot in zip(range(3), image_N0_list, name_plot_list):
#    rec_result_1 = rec_space.element(image_N0[time_itvs // 4])
#    rec_result_2 = rec_space.element(image_N0[time_itvs // 4 * 2])
#    rec_result_3 = rec_space.element(image_N0[time_itvs // 4 * 3])
#    rec_result = rec_space.element(image_N0[time_itvs])
#    rec_proj_data = forward_op(rec_result)
#    plt.figure(index, figsize=(24, 24))
#    plt.subplot(3, 3, 1)
#    plt.imshow(np.rot90(template), cmap='bone',
#               vmin=mini,
#               vmax=maxi)
#    plt.axis('off')
#    #plt.savefig("/home/chchen/SwedenWork_Chong/NumericalResults_S/LDDMM_results/J_V/template_J.png", bbox_inches='tight')
#    plt.colorbar()
#    plt.title(name_plot)
#
#    plt.subplot(3, 3, 2)
#    plt.imshow(np.rot90(rec_result_1), cmap='bone',
#               vmin=mini,
#               vmax=maxi)
#
#    plt.axis('off')
#    plt.colorbar()
#    plt.title('time_pts = {!r}'.format(time_itvs // 4))
#
#    plt.subplot(3, 3, 3)
#    plt.imshow(np.rot90(rec_result_2), cmap='bone',
#               vmin=mini,
#               vmax=maxi)
#    plt.axis('off')
#    plt.colorbar()
#    plt.title('time_pts = {!r}'.format(time_itvs // 4 * 2))
#
#    plt.subplot(3, 3, 4)
#    plt.imshow(np.rot90(rec_result_3), cmap='bone',
#               vmin=mini,
#               vmax=maxi)
#    plt.axis('off')
#    plt.colorbar()
#    plt.title('time_pts = {!r}'.format(time_itvs // 4 * 3))
#
#    plt.subplot(3, 3, 5)
#    plt.imshow(np.rot90(rec_result), cmap='bone',
#               vmin=mini,
#               vmax=maxi)
#    plt.axis('off')
#    plt.colorbar()
#    plt.title('Reconstructed by {!r} iters, '
#        '{!r} projs'.format(niter, num_angles))
#
#    for i in range(nb_data_points):
#        plt.subplot(3, 3, 7 + i)
#        plt.imshow(np.rot90(ground_truth_list[i]), cmap='bone',
#                   vmin=mini,
#                   vmax=maxi)
#        plt.axis('off')
#        plt.colorbar()
#        plt.title('Ground truth time ' + str(data_time_points[i]))
#
#    name=path_result + name_plot + '.png'
#    plt.savefig(name, bbox_inches='tight')
##
#plt.close('all')
