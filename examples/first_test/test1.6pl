#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 15:00:21 2017

@author: bgris
"""


# Define the space the problem should be solved on.
# Here the square [-1, 1] x [-1, 1] discretized on a 100x100 grid.
space = odl.uniform_discr([-1, -1], [1, 1], [100, 100])

# Convolution kernel, a small centered rectangle.
kernel = odl.phantom.cuboid(space, [-0.05, -0.05], [0.05, 0.05])

# Create convolution operator
A = Convolution(kernel)

# Create phantom (the "unknown" solution)
phantom = odl.phantom.shepp_logan(space, modified=True)

# Apply convolution to phantom to create data
g = A(phantom)

# Display the results using the show method
kernel.show('kernel')
phantom.show('phantom')
g.show('convolved phantom')