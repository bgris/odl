#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 12:13:22 2017

@author: bgris
"""

# Copyright 2014-2016 The ODL development group
#
# This file is part of ODL.
#
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


import numpy as np
import odl


# --- Create template and displacement field --- #


# Template space: discretized functions on the rectangle [-1, 1]^2 with
# 100 samples per dimension. Usage of 'linear' interpolation ensures that
# the template gradient is well-defined.
templ_space = odl.uniform_discr([-1, -1], [1, 1], (100, 100), interp='linear')

# The template is a rectangle of size 1.0 x 0.5
template = odl.phantom.cuboid(templ_space, [-0.5, -0.25], [0.5, 0.25])

# Create a product space for displacement field
disp_field_space = templ_space.tangent_bundle


# Define a displacement field that bends the template a bit towards the
# upper left. We use a list of 2 functions and discretize it using the
# disp_field_space.element() method.
sigma = 0.5
disp_func = [
    lambda x: 0.4 * np.exp(-(x[0] ** 2 + x[1] ** 2) / (2 * sigma ** 2)),
    lambda x: -0.3 * np.exp(-(x[0] ** 2 + x[1] ** 2) / (2 * sigma ** 2))]

disp_field = disp_field_space.element(disp_func)

# Show template and displacement field
template.show('Template');
#disp_field.show('Displacement field')


# Initialize the deformation operator with fixed template
deform_op = odl.deform.LinDeformFixedTempl(template)

# Apply the deformation operator to get the deformed template.
deformed_template = deform_op(disp_field)
deformed_template.show('Deformed template');


#%% Tomo


# Make a circular cone beam geometry with flat detector
# Angles: uniformly spaced, n = 360, min = 0, max = pi + fan angle
angle_partition = odl.uniform_partition(0, np.pi + 0.7, 360)
# Detector: uniformly sampled, n = 558, min = -40, max = 40
detector_partition = odl.uniform_partition(-40, 40, 558)
# Geometry with large fan angle
geometry = odl.tomo.FanFlatGeometry(
    angle_partition, detector_partition, src_radius=0.1, det_radius=4)


# Ray transform (= forward projection). We use the ASTRA CUDA backend.
ray_trafo = odl.tomo.RayTransform(templ_space, geometry, impl='astra_cuda')


# Create projection data by calling the ray transform on the template
proj_data = ray_trafo(template)
# Create projection data by calling the ray transform on the deformed template
proj_data_defo = ray_trafo(deformed_template)


proj_data.show(title='Projection data (sinogram)');
proj_data_defo.show(title='Projection deformed data (sinogram)');

#%%
I1=deformed_template
# Creating a function obj_fun which takes in input an image and returns the l2 difference between its ray transform and proj_data
obj_fun = odl.solvers.L2NormSquared(ray_trafo.range) * (ray_trafo - proj_data)
obj_fun(I1)

obj_fun.gradient(I1)

obj_fun.derivative(I1)(template)





















