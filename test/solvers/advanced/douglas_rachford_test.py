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

"""Test for the Douglas-Rachford solver."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import pytest
import odl
from odl.solvers import douglas_rachford_pd

from odl.util.testutils import all_almost_equal, example_element


# Places for the accepted error when comparing results
HIGH_ACCURACY = 8
LOW_ACCURACY = 4


def test_primal_dual_input_handling():
    """Test to see that input is handled correctly."""

    space1 = odl.uniform_discr(0, 1, 10)

    lin_ops = [odl.ZeroOperator(space1), odl.ZeroOperator(space1)]
    prox_cc_g = [odl.solvers.proximal_zero(space1),  # Identity operator
                 odl.solvers.proximal_zero(space1)]  # Identity operator
    prox_f = odl.solvers.proximal_zero(space1)  # Identity operator

    # Check that the algorithm runs. With the above operators, the algorithm
    # returns the input.
    x0 = example_element(space1)
    x = x0.copy()
    niter = 3

    douglas_rachford_pd(x, prox_f, prox_cc_g, lin_ops, tau=1.0,
                        sigma=[1.0, 1.0], niter=niter)

    assert x == x0

    # Testing that sizes needs to agree:
    # Too few sigma_i:s
    with pytest.raises(ValueError):
        douglas_rachford_pd(x, prox_f, prox_cc_g, lin_ops, tau=1.0,
                            sigma=[1.0], niter=niter)

    # Too many operators
    prox_cc_g_too_many = [odl.solvers.proximal_zero(space1),
                          odl.solvers.proximal_zero(space1),
                          odl.solvers.proximal_zero(space1)]
    with pytest.raises(ValueError):
        douglas_rachford_pd(x, prox_f, prox_cc_g_too_many, lin_ops,
                            tau=1.0, sigma=[1.0, 1.0], niter=niter)

    # Test for correct space
    space2 = odl.uniform_discr(1, 2, 10)
    x = example_element(space2)
    with pytest.raises(ValueError):
        douglas_rachford_pd(x, prox_f, prox_cc_g, lin_ops, tau=1.0,
                            sigma=[1.0, 1.0], niter=niter)


def test_primal_dual_l1():
    """Verify that the correct value is returned for l1 dist optimization.

    Solves the optimization problem

        min_x ||x - data_1||_1 + 0.5 ||x - data_2||_1

    which has optimum value data_1.
    """

    # Define the space
    space = odl.rn(5)

    # Operator
    L = [odl.IdentityOperator(space)]

    # Data
    data_1 = odl.util.testutils.example_element(space)
    data_2 = odl.util.testutils.example_element(space)

    # Proximals
    prox_f = odl.solvers.proximal_l1(space, g=data_1)
    prox_cc_g = [odl.solvers.proximal_cconj_l1(space, g=data_2, lam=0.5)]

    # Solve with f term dominating
    x = space.zero()
    douglas_rachford_pd(x, prox_f, prox_cc_g, L,
                        tau=3.0, sigma=[1.0], niter=10)

    assert all_almost_equal(x, data_1, places=2)


def test_primal_dual_with_li():
    """Test for the forward-backward solver with infimal convolution.

    The test is done by minimizing the functional ``(g @ l)(x)``, where

        ``(g @ l)(x) = inf_y { g(y) + l(x - y) }``,

    g is the indicator function on [-3, -1], and l(x) = 1/2||x||_2^2.
    The optimal solution to this problem is given by x in [-3, -1].
    """
    # Parameter values for the box constraint
    upper_lim = -1
    lower_lim = -3

    space = odl.rn(1)

    lin_op = odl.IdentityOperator(space)
    lin_ops = [lin_op]
    prox_cc_g = [odl.solvers.proximal_cconj(
                 odl.solvers.proximal_box_constraint(space,
                                                     lower=lower_lim,
                                                     upper=upper_lim))]
    prox_f = odl.solvers.proximal_zero(space)

    prox_cc_ls = [odl.solvers.proximal_cconj_l2_squared(space)]

    # Centering around a point further away from [-3,-1].
    x = space.element(10)

    douglas_rachford_pd(x, prox_f, prox_cc_g, lin_ops, tau=0.5,
                        sigma=[1.0], niter=20, prox_cc_l=prox_cc_ls)

    assert lower_lim - 10 ** -LOW_ACCURACY <= x[0]
    assert x[0] <= upper_lim + 10 ** -LOW_ACCURACY


if __name__ == '__main__':
    pytest.main(str(__file__.replace('\\', '/') + ' -v'))
