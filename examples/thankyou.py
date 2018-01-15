#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 12:23:55 2018

@author: barbara
"""


import odl
import numpy as np
from matplotlib import pylab as plt


import function_generate_data_doigt_bis as fun_gen

#
#%%


# Discrete reconstruction space: discretized functions on the rectangle
rec_space = odl.uniform_discr(
    min_pt=[-16, -16], max_pt=[16, 16], shape=[512, 512],
    dtype='float32', interp='linear')



#%%
miny = -10
maxy = 10
width = 1


at0 = [miny, ]