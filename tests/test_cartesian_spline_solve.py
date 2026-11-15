r"""`InterpolatedField_Cartesian` reproduces a field that its splines can represent
exactly.

A spline of degree :math:`k` reproduces polynomials of degree :math:`\le k`, so
on any grid a quintic (or cubic) spline of a quadratic :math:`\psi` and a
linear :math:`\mathbf{B}` is exact: the only error left is in solving for the
spline coefficients. scipy (>= 1.13) solves for them iteratively, and its
default stops at a residual of 1e-5 times the norm of all the grid values.
The coefficient errors that leaves are amplified by the second derivatives of
:math:`\psi`, which enter the beam equation: on this grid, with scipy's
default, they are off by up to 0.012 (quintic) and 0.009 (cubic) against
values of order 1 (scipy 1.18). Without
`geometry.CUBOID_SPLINE_SOLVER_ARGS`, every test here fails.

The derivatives are Scotty's own (finite differences of the spline, step
1e-4 m), which are exact for a quadratic up to rounding, ~1e-16 / 1e-8.
"""

import numpy as np
import pytest

from scotty.geometry import InterpolatedField_Cartesian

# psi = A (X - X0)^2 + B Y^2 + C Z^2 + D X Z + E X Y + F, and its exact Hessian
A, B, C, D, E, F, X0 = 0.8, 0.5, 1.2, 0.3, -0.2, 0.1, 1.5
HESSIAN = {
    "XX": 2 * A,
    "YY": 2 * B,
    "ZZ": 2 * C,
    "XY": E,
    "XZ": D,
    "YZ": 0.0,
}
DELTA = 1e-4


def psi(X, Y, Z):
    return A * (X - X0) ** 2 + B * Y**2 + C * Z**2 + D * X * Z + E * X * Y + F


def gradient(X, Y, Z):
    return (
        2 * A * (X - X0) + D * Z + E * Y,
        2 * B * Y + E * X,
        2 * C * Z + D * X,
    )


def B_field(X, Y, Z):
    """Linear in each coordinate (not divergence-free; only the interpolation is tested)"""
    return 0.1 + 0.2 * Y - 0.1 * Z, 1.0 + 0.3 * X, 0.05 - 0.4 * X + 0.1 * Y


@pytest.fixture(scope="module", params=["quintic", "cubic"])
def field(request):
    X = np.linspace(1.0, 2.0, 26)
    Y = np.linspace(-0.4, 0.4, 21)
    Z = np.linspace(-0.5, 0.5, 26)
    G = np.meshgrid(X, Y, Z, indexing="ij")
    return InterpolatedField_Cartesian(
        X, Y, Z, *B_field(*G), psi(*G), interp_order=request.param
    )


@pytest.fixture(scope="module")
def points():
    """Points inside the grid, away from its edges"""
    rng = np.random.default_rng(0)
    return (
        rng.uniform(1.1, 1.9, 200),
        rng.uniform(-0.3, 0.3, 200),
        rng.uniform(-0.4, 0.4, 200),
    )


def test_polflux_and_B_are_exact(field, points):
    X, Y, Z = points
    np.testing.assert_allclose(field.polflux(X, Y, Z), psi(X, Y, Z), rtol=0, atol=1e-10)
    for got, want in zip((field.B_X, field.B_Y, field.B_Z), B_field(X, Y, Z)):
        np.testing.assert_allclose(got(X, Y, Z), want, rtol=0, atol=1e-10)


def test_polflux_gradient(field, points):
    X, Y, Z = points
    got = (
        field.d_polflux_dX(X, Y, Z, DELTA),
        field.d_polflux_dY(X, Y, Z, DELTA),
        field.d_polflux_dZ(X, Y, Z, DELTA),
    )
    for g, w in zip(got, gradient(X, Y, Z)):
        np.testing.assert_allclose(g, w, rtol=0, atol=1e-7)


def test_polflux_hessian(field, points):
    """The terms that enter the beam equation through grad grad H. Before the
    solver tolerances were set, these were off by up to 0.012 here"""
    X, Y, Z = points
    got = {
        "XX": field.d2_polflux_dX2(X, Y, Z, DELTA),
        "YY": field.d2_polflux_dY2(X, Y, Z, DELTA),
        "ZZ": field.d2_polflux_dZ2(X, Y, Z, DELTA),
        "XY": field.d2_polflux_dXdY(X, Y, Z, DELTA, DELTA),
        "XZ": field.d2_polflux_dXdZ(X, Y, Z, DELTA, DELTA),
        "YZ": field.d2_polflux_dYdZ(X, Y, Z, DELTA, DELTA),
    }
    for k, want in HESSIAN.items():
        # rounding in a second difference with step 1e-4: ~1e-16 * |psi| / 1e-8
        np.testing.assert_allclose(got[k], want, rtol=0, atol=1e-5, err_msg=k)
