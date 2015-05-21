# -*- coding: utf-8 -*-
"""
simple_test_astra.py -- a simple test script

Copyright 2014, 2015 Holger Kohr

This file is part of RL.

RL is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

RL is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with RL.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import division, print_function, unicode_literals, absolute_import
from future import standard_library
standard_library.install_aliases()
from math import sin,cos,pi

import numpy as np
import RL.operator.operator as OP
import RL.space.function as fs
import RL.space.cuda as cs
import RL.space.euclidean as ds
import RL.space.product as prod
import RL.space.discretizations as dd
import RL.space.set as sets
import SimRec2DPy as SR
import RL.operator.solvers as solvers

import matplotlib.pyplot as plt


class ProjectionGeometry(object):
    """ Geometry for a specific projection
    """
    def __init__(self, sourcePosition, detectorOrigin, pixelDirection):
        self.sourcePosition = sourcePosition
        self.detectorOrigin = detectorOrigin
        self.pixelDirection = pixelDirection

class CudaProjector(OP.LinearOperator):
    """ A projector that creates several projections as defined by geometries
    """
    def __init__(self, volumeOrigin, voxelSize, nVoxels, nPixels, stepSize, geometries, domain, range):
        self.geometries = geometries
        self.domain = domain
        self.range = range
        self.forward = SR.SRPyCuda.CudaForwardProjector(nVoxels, volumeOrigin, voxelSize, nPixels, stepSize)
        self._adjoint = CudaBackProjector(volumeOrigin, voxelSize, nVoxels, nPixels, stepSize, geometries, range, domain)

    def _apply(self, data, out):
        #Create projector
        self.forward.setData(data.data_ptr)

        #Project all geometries
        for i in range(len(self.geometries)):
            geo = self.geometries[i]
            self.forward.project(geo.sourcePosition, geo.detectorOrigin, geo.pixelDirection, out[i].data_ptr)

    @property
    def adjoint(self):
        return self._adjoint


class CudaBackProjector(OP.LinearOperator):
    def __init__(self, volumeOrigin, voxelSize, nVoxels, nPixels, stepSize, geometries, domain, range):
        self.geometries = geometries
        self.domain = domain
        self.range = range
        self.back = SR.SRPyCuda.CudaBackProjector(nVoxels, volumeOrigin, voxelSize, nPixels, stepSize)

    def _apply(self, projections, out):
        #Zero out the return data
        out.set_zero()

        #Append all projections
        for i in range(len(self.geometries)):
            geo = self.geometries[i]
            self.back.backProject(geo.sourcePosition, geo.detectorOrigin, geo.pixelDirection, projections[i].data_ptr, out.data_ptr)



#Set geometry parameters
volumeSize = np.array([20.0,20.0])
volumeOrigin = -volumeSize/2.0

detectorSize = 50.0
detectorOrigin = -detectorSize/2.0

sourceAxisDistance = 20.0
detectorAxisDistance = 20.0

#Discretization parameters
nVoxels = np.array([500, 500])
nPixels = 4000
nProjection = 1000

#Scale factors
voxelSize = volumeSize/nVoxels
pixelSize = detectorSize/nPixels
stepSize = voxelSize.max()

#Define projection geometries
geometries = []
for theta in np.linspace(0, 2*pi, nProjection):
    x0 = np.array([cos(theta), sin(theta)])
    y0 = np.array([-sin(theta), cos(theta)])

    projSourcePosition = -sourceAxisDistance * x0
    projDetectorOrigin = detectorAxisDistance * x0 + detectorOrigin * y0
    projPixelDirection = y0 * pixelSize
    geometries.append(ProjectionGeometry(projSourcePosition, projDetectorOrigin, projPixelDirection))

#Define the space of one projection
projectionSpace = fs.L2(sets.Interval(0, detectorSize))
projectionRN = cs.CudaRN(nPixels)

#Discretize projection space
projectionDisc = dd.makeUniformDiscretization(projectionSpace, projectionRN)

#Create the data space, which is the Cartesian product of the single projection spaces
dataDisc = prod.powerspace(projectionDisc, nProjection)

#Define the reconstruction space
reconSpace = fs.L2(sets.Rectangle([0, 0], volumeSize))

#Discretize the reconstruction space
reconRN = cs.CudaRN(nVoxels.prod())
reconDisc = dd.makePixelDiscretization(reconSpace, reconRN, nVoxels[0], nVoxels[1])

#Create a phantom
phantom = SR.SRPyUtils.phantom(nVoxels)
phantomVec = reconDisc.element(phantom)

#Make the operator
projector = CudaProjector(volumeOrigin, voxelSize, nVoxels, nPixels, stepSize, geometries, reconDisc, dataDisc)

#Apply once to find norm estimate
projections = projector(phantomVec)
recon = projector.T(projections)
normEst = recon.norm() / phantomVec.norm()

#Define function to plot each result
plt.figure()
plt.ion()
plt.set_cmap('bone')
def plotResult(x):
    plt.imshow(x[:].reshape(nVoxels))
    plt.draw()
    print((x-phantomVec).norm())
    plt.pause(0.01)

#Solve using landweber
x = reconDisc.zero()
#solvers.landweber(projector, x, projections, 200, omega=0.4/normEst, part_results=solvers.ForEachPartial(plotResult))
solvers.landweber(projector, x, projections, 5, omega=0.4/normEst, part_results=solvers.PrintIterationPartial())
#solvers.conjugate_gradient(projector, x, projections, 20, part_results=solvers.ForEachPartial(plotResult))

plt.imshow(x[:].reshape(nVoxels))
plt.show()