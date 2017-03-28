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

# Need operator norm for step length (omega)
opnorm = odl.power_method_opnorm(A)

f = space.zero()
odl.solvers.landweber(A, f, g, niter=100, omega=1/opnorm**2)
f.show('landweber')

f = space.zero()
odl.solvers.conjugate_gradient_normal(A, f, g, niter=100)
f.show('conjugate gradient')
B = odl.IdentityOperator(space)
a = 0.1
T = A.adjoint * A + a * B.adjoint * B
b = A.adjoint(g)

f = space.zero()
odl.solvers.conjugate_gradient(T, f, b, niter=100)
f.show('Tikhonov identity conjugate gradient')

B = odl.IdentityOperator(space)
a = 0.1
T = A.adjoint * A + a * B.adjoint * B
b = A.adjoint(g)

f = space.zero()
odl.solvers.conjugate_gradient(T, f, b, niter=100)
f.show('Tikhonov identity conjugate gradient')

B = odl.Gradient(space)
a = 0.0001
T = A.adjoint * A + a * B.adjoint * B
b = A.adjoint(g)

f = space.zero()
odl.solvers.conjugate_gradient(T, f, b, niter=100)
f.show('Tikhonov gradient conjugate gradient')

# Assemble all operators into a list.
grad = odl.Gradient(space)
lin_ops = [A, grad]
a = 0.001

# Create functionals for the l2 distance and l1 norm.
g_funcs = [odl.solvers.L2NormSquared(space).translated(g),
           a * odl.solvers.L1Norm(grad.range)]

# Functional of the bound constraint 0 <= x <= 1
f = odl.solvers.IndicatorBox(space, 0, 1)

# Find scaling constants so that the solver converges.
# See the douglas_rachford_pd documentation for more information.
opnorm_A = odl.power_method_opnorm(A, xstart=g)
opnorm_grad = odl.power_method_opnorm(grad, xstart=g)
sigma = [1 / opnorm_A ** 2, 1 / opnorm_grad ** 2]
tau = 1.0

# Solve using the Douglas-Rachford Primal-Dual method
x = space.zero()
odl.solvers.douglas_rachford_pd(x, f, g_funcs, lin_ops,
                                tau=tau, sigma=sigma, niter=100)
x.show('TV Douglas-Rachford')