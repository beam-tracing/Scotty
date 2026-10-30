"""The sum-of-squares Booker discriminant (fun_general.find_Booker_discriminant).

It must equal beta**2 - 4*alpha*gamma as a polynomial, never be negative, and
remove the rounding noise that the cancelled form puts into H near zero
density. The last point is what lets small finite-difference steps work at
the plasma edge, and it is why the K-Hessian of H in vacuum comes out as the
exact 2/K0**2 on the diagonal.
"""

from scotty.fun_general import (
    find_Booker_discriminant,
    find_Booker_terms,
    freq_GHz_to_angular_frequency,
)
from scotty.geometry import CircularFieldBase
from scotty.hamiltonian import Hamiltonian, hessians
from scotty.logger import config_logger
from scotty.profile_fit import QuadraticFit

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import constants

OMEGA = freq_GHz_to_angular_frequency(55.0)


@pytest.fixture(scope="module", autouse=True)
def trace_level(tmp_path_factory):
    config_logger("warning", None, tmp_path_factory.mktemp("log"), "")


def test_sum_of_squares_equals_cancelled_form():
    """Random physical points above and below the cut-offs, including |Y| > 1.
    The two forms agree to rounding, measured against the size of the terms
    being cancelled"""
    rng = np.random.default_rng(0)
    n = 20000
    B = rng.uniform(0.1, 6.0, n)
    density = 10 ** rng.uniform(-7, 1.5, n)  # X from ~1e-7 to ~30
    sin_sq = rng.uniform(0.0, 1.0, n)

    alpha, beta, gamma = find_Booker_terms(OMEGA, B, sin_sq, density)
    cancelled = beta**2 - 4 * alpha * gamma
    stable = find_Booker_discriminant(OMEGA, B, sin_sq, density)

    scale = beta**2 + np.abs(4 * alpha * gamma)
    assert np.max(np.abs(stable - cancelled) / scale) < 1e-13
    assert np.all(stable >= 0)


def test_vacuum_limit():
    """At zero density alpha = 1, beta = -2, gamma = 1, so the discriminant is 0
    and H = (K/K0)^2 - 1. The stable form gives exactly 0; the cancelled form
    gives 0 only up to rounding"""
    assert find_Booker_discriminant(OMEGA, 1.0, 0.3, 0.0) == 0.0


class ToroidalField(CircularFieldBase):
    def __init__(self, R0=1.5, a=0.5, B0=1.0):
        self.R0, self.a, self.B0 = R0, a, B0
        self.R_axis, self.minor_radius_a = R0, a

    def B_R(self, R, _, Z):
        return np.zeros_like(R * Z, dtype=float)

    def B_T(self, R, _, Z):
        return self.B0 * self.R0 / R * np.ones_like(Z, dtype=float)

    def B_Z(self, R, _, Z):
        return np.zeros_like(R * Z, dtype=float)


def _hamiltonian(mode_flag, delta_q=1e-5, delta_K=0.1):
    return Hamiltonian(
        launch_angular_freq=OMEGA,
        mode_flag=mode_flag,
        deltas=np.array([delta_q, delta_q, delta_K, delta_K, delta_K]),
        field=ToroidalField(),
        density_fit=QuadraticFit(1.0, 4.0),
    )


@pytest.mark.parametrize("mode_flag", (1, -1))
def test_noise_floor_of_H_near_zero_density(mode_flag):
    """Just inside the plasma edge (X ~ 3e-4), the std of H about a quadratic
    over 41 points 1e-12 m apart is rounding, not structure. With the cancelled
    discriminant it grows like 1/X (about 1e-12 here, 1e-9 at X ~ 1e-10); it
    must now sit near 1e-16"""
    H = _hamiltonian(mode_flag)
    # n_e = 4 (1 - rho^2) * 1e19 m^-3, so rho = 0.99985 gives X ~ 3e-4 at 55 GHz
    R0 = 1.5 + 0.5 * 0.99985
    j = np.arange(-20, 21)
    R = R0 + j * 1e-12
    K_R, K_zeta, K_Z = -1000.0, 50.0, 100.0
    values = np.array([H(r, 0.0, 0.0, K_R, K_zeta, K_Z) for r in R])
    residual = values - np.polyval(np.polyfit(j, values, 2), j)

    assert np.std(residual) < 1e-14


@pytest.mark.parametrize("mode_flag", (1, -1))
def test_K_hessian_in_vacuum(mode_flag):
    """Just inside the plasma edge (X ~ 1e-6) the Booker term is a constant
    to O(X), so grad_K grad_K H is 2/K0^2 on the K_R and K_Z diagonals,
    2/(K0^2 R^2) for K_zeta (a mode number), and zero off the diagonal.
    At exactly zero density even the cancelled discriminant is exact
    (4 - 4 = 0); the rounding noise appears at small nonzero X, which is
    where the beam solver starts. There, with the cancelled form and the
    default steps, the finite differences gave the wrong sign on the
    diagonal and spurious off-diagonal terms"""
    H = _hamiltonian(mode_flag, delta_q=1e-4, delta_K=0.1)
    # n_e = 4 (1 - rho^2) * 1e19 m^-3, so rho = 0.9999995 gives X ~ 1e-6 at 55 GHz;
    # the O(X) correction to the Hessian is then far below rtol
    R = 1.5 + 0.5 * 0.9999995
    K0 = OMEGA / constants.c
    dH = H.derivatives(np.array([R, 0.0, 0.0]), np.array([-1000.0, 50.0, 100.0]), second_order=True)
    _, _, gradK_gradK_H = hessians(dH, cartesian=False)

    expected = np.diag([2 / K0**2, 2 / (K0**2 * R**2), 2 / K0**2])
    assert_allclose(gradK_gradK_H, expected, rtol=1e-4, atol=1e-10)
