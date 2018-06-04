#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 16:37:59 2017

@author: bgris
"""
# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
import numpy as np
import matplotlib.pyplot as plt
import odl
from odl.discr import uniform_discr, Gradient
from odl.phantom import sphere, sphere2, cube

from odl.tomo import Parallel3dAxisGeometry

import odl.deform.mrc_data_io
from odl.discr import uniform_partition

from odl.deform.mrc_data_io import (read_mrc_data, geometry_mrc_data,
                                    result_2_mrc_format, result_2_nii_format)
from odl.contrib.mrc import FileReaderMRC

from odl.tomo import RayTransform, fbp_op
from odl.operator import (BroadcastOperator, power_method_opnorm)
from odl.solvers import (CallbackShow, CallbackPrintIteration, ZeroFunctional,
                         L2NormSquared, L1Norm, SeparableSum,
                         chambolle_pock_solver, conjugate_gradient_normal)
standard_library.install_aliases()

# Get the path of data cormack
directory = '/home/bgris/data/TEM/RNA_Polymerase_II_One_particle/'

data_filename = 'tiltseries_nonoise.mrc'
file_path = directory + data_filename
data, data_extent, header= read_mrc_data(file_path=file_path,
                                                           normalize=False)
#data_extent=np.moveaxis(data_extent, -1, 0).copy()


with FileReaderMRC(file_path) as data_reader:
  data_header,data = data_reader.read()


angle_partition=uniform_partition(-np.radians(60),np.radians(60),61)
data_shape=data.shape
detector_partition = uniform_partition(-data_extent[0:2] / 2.0 ,
                                               data_extent[0:2] / 2.0 ,
                                               data_shape[0:2])
#data_shape=data.shape
#detector_partition = uniform_partition([-160,-160],
#                                                [160,160],
#                                               data_shape[0:2])

data=np.moveaxis(data, -1, 0).copy()

single_axis_geometry=Parallel3dAxisGeometry(angle_partition, detector_partition,
                                            axis=[1,0,0])


phantom_filename='rna_phantom.mrc'
phantom_file_path = directory + phantom_filename

phantom, phantom_extent, phantomheader= read_mrc_data(file_path=phantom_file_path,
                                                           normalize=False)

rec_shape = (phantom.shape[0], phantom.shape[1], phantom.shape[2])

rec_extent = np.asarray(rec_shape, float)

vox_size=0.5
# Reconstruction space with physical setting
rec_space = uniform_discr(-rec_extent / 2 * vox_size,
                          rec_extent / 2  * vox_size, rec_shape,
dtype='float32', interp='linear')
phantom=rec_space.element(phantom)
# --- Creating forward operator --- #

# Create forward operator
forward_op = RayTransform(rec_space, single_axis_geometry, impl='astra_cuda')

result = forward_op(phantom)
#result /= np.linalg.norm(data) / np.linalg.norm(result)
#(result-np.moveaxis(data, -1, 0)).show()


result.show('phantom 0', coords=[0, None, None])
forward_op.range.element(data).show('data 0', coords=[0, None, None])
#forward_op.range.element(np.moveaxis(data, -1, 0)).show('data 0', coords=[0, None, None])



result.show('phantom 1',coords=[ None, None,0])
forward_op.range.element(data).show('data 1',coords=[ None, None,0])



result.show('phantom 2',coords=[ None, None,0])
forward_op.range.element(data).show('data 2',coords=[ None, None,0])
#forward_op.range.element(np.moveaxis(data, -1, 0)).show('data 2',coords=[ None, None,0])

#%%

template=phantom.copy()

