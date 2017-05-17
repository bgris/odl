#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 11:31:20 2017

@author: bgris
"""

# ODL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ODL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ODL.  If not, see <http://www.gnu.org/licenses/>.

"""
ET image reconstruction using LDDMM.
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
import numpy as np
import matplotlib.pyplot as plt
import odl
from odl.discr import uniform_discr, Gradient
from odl.phantom import sphere, sphere2, cube

from odl.deform.mrc_data_io import (read_mrc_data, geometry_mrc_data,
                                    result_2_mrc_format, result_2_nii_format)
from odl.tomo import RayTransform, fbp_op
from odl.operator import (BroadcastOperator, power_method_opnorm)
from odl.solvers import (CallbackShow, CallbackPrintIteration, ZeroFunctional,
                         L2NormSquared, L1Norm, SeparableSum,
                         chambolle_pock_solver, conjugate_gradient_normal)
standard_library.install_aliases()


#%%%
# --- Reading data --- #

# Get the path of data
directory = '/home/bgris/odl/examples/TEM/wetransfer-569840/'
data_filename = 'triangle_crop.mrc'
file_path = directory + data_filename
data, data_extent, header, extended_header = read_mrc_data(file_path=file_path,
                                                           force_type='FEI1',
                                                           normalize=True)

#Downsample the data
downsam = 1
data_downsam = data[:, :, ::downsam]

# --- Getting geometry --- #
det_pix_size = 0.521

# Create 3-D parallel projection geometry
single_axis_geometry = geometry_mrc_data(data_extent=data_extent,
                                         data_shape=data.shape,
                                         det_pix_size=det_pix_size,
                                         units='physical',
                                         extended_header=extended_header,
                                         downsam=downsam)
