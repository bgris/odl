#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 14:19:59 2018

@author: bgris
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 12:32:43 2018

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

niter=300
eps=0.002
## Give regularization parameter
lamb = 1e3
name_lamb='1e_' + str(-int(np.log(lamb)/np.log(10)))




name_list_template = ['SheppLogan0', 'SheppLogan4']
name_list_ground_truth = ['SheppLogan4_deformed']
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


name_exp_result = 'RecoL2diff' + name_val + 'num_angles_' + str(num_angles) + '_min_angle_0_max_angle_'
name_exp_result += maxiangle + '_noise_' + noi


path_data = '/home/' + namepath + '/Dropbox/Recherche/mes_publi/Metamorphosis_PDE_ODE/data/test6/'
path_result_init = '/home/' + namepath + '/Results/Metamorphosis/test6/'
#path_result_init = '/home/bgris/Dropbox/Recherche/mes_publi/Metamorphosis_PDE_ODE/Results/test2/'
path_result = path_result_init + name_exp_result  + '__lamb_'
path_result += name_lamb  + '/'



path_result_init_dropbox = '/home/' + namepath + '/Dropbox/Recherche/mes_publi/Metamorphosis_PDE_ODE/Results_ODE/test6/'
path_result_dropbox = path_result_init_dropbox + name_exp_result +'__lamb_'
path_result_dropbox += name_lamb + '/'



# Discrete reconstruction space: discretized functions on the rectangle
rec_space = odl.uniform_discr(
    min_pt=[-16, -16], max_pt=[16, 16], shape=[256, 256],
    dtype='float32', interp='linear')


name_ground_truth = path_data + name_val
ground_truth = rec_space.element(np.loadtxt(name_ground_truth))

name_template = path_data + name_val_template
template = rec_space.element(np.loadtxt(name_template))




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



data=data_load
#data=[proj_data]
data_time_points=np.array([1])
forward_operators=[forward_op]
Data_discrepancy=odl.solvers.L2NormSquared(forward_op.range).translated(data)
#Regularizer=odl.solvers.L2NormSquared(rec_space)

##%%

gradient = odl.Gradient(rec_space)
# Functional for TV minimization

#template= rec_space.zero()

cross_norm = (1/lamb) * odl.solvers.GroupL1Norm(gradient.range).translated(gradient(template))

# --- Create functionals for solving the optimization problem ---

# Assemble operators and functionals for the solver
lin_ops = [forward_op, gradient]
g = [Data_discrepancy, cross_norm]

# Create callback that prints the iteration number and shows partial results
callback = (odl.solvers.CallbackShow('iterates', step=5, clim=[0, 1]) &
            odl.solvers.CallbackPrintIteration())

f = odl.solvers.ZeroFunctional(rec_space)
# odl.solvers.IndicatorNonnegativity(rec_space)

# Solve with initial guess x = 0.
# Step size parameters are selected to ensure convergence.
# See douglas_rachford_pd doc for more information.
x = forward_op.domain.zero()
odl.solvers.douglas_rachford_pd(x, f, g, lin_ops,
                                tau=0.1, sigma=[0.1, 0.02], lam=1.5,
                                niter=500, callback=callback)




#%%




##%%
mini=-1
maxi=1
#
os.mkdir(path_result)
os.mkdir(path_result_dropbox)

np.savetxt(path_result + 'reco' , X)


X.show(clim = [mini, maxi])
name=path_result_dropbox + 'reco' + '.png'
plt.savefig(name, bbox_inches='tight')




plt.close('all')