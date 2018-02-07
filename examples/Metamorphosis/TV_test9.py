#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 18:13:41 2018

@author: bgris
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 10:48:00 2018

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

index_angle = 4
index_maxangle = 0
index_minangle = 0
index_noise = 2

## The parameter for kernel function
sigma = 3.0
name_sigma=str(int(sigma))

niter=300
epsV=0.02
epsZ=0.02
## Give regularization parameter
lamb = 1e-5
name_lamb='1e_' + str(-int(np.log(lamb)/np.log(10)))
tau = 1e-6
name_tau='1e_' + str(-int(np.log(tau)/np.log(10)))

# Give the number of time points
time_itvs = 20
nb_time_point_int=time_itvs




name_list_template = ['SheppLogan10']
name_list_ground_truth = ['SheppLogan11_deformed']
num_angles_list = [10, 50, 100, 20, 30]
maxiangle_list = ['pi', '0_25pi', '0_5pi', '0_75pi']
max_angle_list = [np.pi, 0.25*np.pi, 0.5*np.pi, 0.75*np.pi]
noise_level_list = [0.0, 0.05, 0.25]
noi_list = ['0', '0_05', '0_25']
miniangle_list = ['0', '0_25pi']
min_angle_list = [0, 0.25*np.pi]
name_val_template = name_list_template[index_name_template]
name_val = name_list_ground_truth[index_name_ground_truth]
num_angles = num_angles_list[index_angle]
maxiangle = maxiangle_list[index_maxangle]
max_angle = max_angle_list[index_maxangle]
noise_level = noise_level_list[index_noise]
noi = noi_list[index_noise]
min_angle = min_angle_list[index_minangle]
miniangle = miniangle_list[index_minangle]

name_exp = name_val + 'num_angles_' + str(num_angles) + '_min_angle_' + miniangle + '_max_angle_'
name_exp += maxiangle + '_noise_' + noi


path_data = '/home/' + namepath + '/data/Metamorphosis/test11/'
path_result_init = '/home/' + namepath + '/Results/Metamorphosis/test11/'
#path_result_init = '/home/bgris/Dropbox/Recherche/mes_publi/Metamorphosis_PDE_ODE/Results/test2/'
path_result = path_result_init + name_exp + '__sigma_' + name_sigma + '__lamb_'
path_result += name_lamb + '__tau_' + name_tau + '__niter_' + str(niter) + '__ntimepoints_' + str(time_itvs) + '/'



#path_result_init_dropbox = '/home/' + namepath + '/Dropbox/Recherche/mes_publi/Metamorphosis_PDE_ODE/Results_ODE/test8/'
#path_result_dropbox = path_result_init_dropbox + name_exp + '__sigma_' + name_sigma + '__lamb_'
#path_result_dropbox += name_lamb + '__tau_' + name_tau + '__niter_' + str(niter) + '__ntimepoints_' + str(time_itvs) + '/'



# Discrete reconstruction space: discretized functions on the rectangle
space = odl.uniform_discr(
    min_pt=[-16, -16], max_pt=[16, 16], shape=[256, 256],
    dtype='float32', interp='linear')


name_ground_truth = path_data + name_val
ground_truth = space.element(np.loadtxt(name_ground_truth))

name_template = path_data + name_val_template
template = space.element(np.loadtxt(name_template))




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
ray_trafo = odl.tomo.RayTransform(space, geometry, impl='astra_cpu')


## load data

data_load = ray_trafo.range.element(np.loadtxt(path_data + name_exp))



data=data_load.copy()
#data=[proj_data]


# --- Create functionals for solving the optimization problem ---

# Gradient for TV regularization
gradient = odl.Gradient(space)

# Functional to enforce 0 <= x <= 1
f = odl.solvers.IndicatorBox(space, -1.5, 1.5)

lam = 1
data_matching_list = ['exact', 'inexact']
for data_matching in data_matching_list :
    if data_matching == 'exact':
        # Functional to enforce Ax = g
        # Due to the splitting used in the douglas_rachford_pd solver, we only
        # create the functional for the indicator function on g here, the forward
        # model is handled separately.
        indicator_zero = odl.solvers.IndicatorZero(ray_trafo.range)
        indicator_data = indicator_zero.translated(data)
    elif data_matching == 'inexact':
        # Functional to enforce ||Ax - g||_2 < eps
        # We do this by rewriting the condition on the form
        # f(x) = 0 if ||A(x/eps) - (g/eps)||_2 < 1, infinity otherwise
        # That function (with A handled separately, as mentioned above) is
        # implemented in ODL as the IndicatorLpUnitBall function.
        # Note that we use right multiplication in order to scale in input argument
        # instead of the result of the functional, as would be the case with left
        # multiplication.
        eps = 5.0
    
        # Add noise to data
        raw_noise = odl.phantom.white_noise(ray_trafo.range)
        data += raw_noise * eps / raw_noise.norm()
    
        # Create indicator
        indicator_l2_ball = odl.solvers.IndicatorLpUnitBall(ray_trafo.range, 2)
        indicator_data = indicator_l2_ball.translated(data / eps) * (1 / eps)
    else:
        raise RuntimeError('unknown data_matching')
    
    # Functional for TV minimization
    cross_norm = lam * odl.solvers.GroupL1Norm(gradient.range)
    
    # --- Create functionals for solving the optimization problem ---
    
    # Assemble operators and functionals for the solver
    lin_ops = [ray_trafo, gradient]
    g = [indicator_data, cross_norm]
    
    # Create callback that prints the iteration number and shows partial results
    callback = (odl.solvers.CallbackShow('iterates', step=5, clim=[0, 1]) &
                odl.solvers.CallbackPrintIteration())
    
    # Solve with initial guess x = 0.
    # Step size parameters are selected to ensure convergence.
    # See douglas_rachford_pd doc for more information.
    x = ray_trafo.domain.zero()
    odl.solvers.douglas_rachford_pd(x, f, g, lin_ops,
                                    tau=0.1, sigma=[0.1, 0.02], lam=1.5,
                                    niter=500, callback=callback)
    
    np.savetxt(path_result + '_TV_' + data_matching + 'num_angles_' + str(num_angles) + '__lam_' + str(lam), x)


# Compare with filtered back-projection
#fbp_recon = odl.tomo.fbp_op(ray_trafo)(data)
#fbp_recon.show('FBP reconstruction')
#phantom.show('Phantom')
#data.show('Sinogram')