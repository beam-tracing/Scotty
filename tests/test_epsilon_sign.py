"""The sign of epsilon_12 (`epsilon_g`).

With fields ~ exp(-i Omega t) and b_hat = B / |B|, the electrons (charge -e)
gyrate in the positive sense about b_hat, so of the two circularly polarised
waves along B, the one rotating with them, u1 + i u2, is the one that
resonates at Y = |Omega_ce| / Omega -> 1: N^2 = 1 - X / (1 - Y). Scotty keeps
the charge unsigned (Y > 0), so epsilon_12 = -X Y / (1 - Y^2), negative for
Y < 1. Hall-Chen, Parra & Hillesheim (2022) eqs. (2) and (35) have the
opposite sign.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import constants

from scotty.fun_general import (
    find_epsilon_terms,
    find_H_Cardano_eigh,
    find_normalised_gyro_ang_freq,
    find_normalised_plasma_ang_freq,
)
from scotty.hamiltonian import DielectricTensor

OMEGA = 2 * np.pi * 55e9
B = 1.0  # T, so Y = 0.51
N_E = 0.5  # 10^19 m^-3, so X = 0.13


def X_and_Y():
    X = find_normalised_plasma_ang_freq(OMEGA, N_E) ** 2
    Y = find_normalised_gyro_ang_freq(OMEGA, B)
    return X, Y


def test_epsilon_12_is_negative_with_the_charge_unsigned():
    """The formula, in both places it is computed (a regression check; the
    physics is in the test below)"""
    X, Y = X_and_Y()
    assert 0 < Y < 1 and 0 < X < 1 - Y
    expected = -X * Y / (1 - Y**2)
    assert_allclose(DielectricTensor(OMEGA, B, N_E).e_12, expected, rtol=1e-12)
    assert_allclose(find_epsilon_terms(OMEGA, B, N_E)[2], expected, rtol=1e-12)


@pytest.mark.parametrize(
    "rotating_with_electrons, N_sq",
    [
        (True, lambda X, Y: 1 - X / (1 - Y)),
        (False, lambda X, Y: 1 - X / (1 + Y)),
    ],
)
def test_circular_polarisation_along_B(rotating_with_electrons, N_sq):
    """K along b_hat (theta_m = 90 degrees): the root of D at N^2 = 1 - X/(1 -
    Y) has polarisation u1 + i u2, rotating with the electrons, and the root
    at N^2 = 1 - X/(1 + Y) has u1 - i u2"""
    X, Y = X_and_Y()
    K = np.sqrt(N_sq(X, Y)) * OMEGA / constants.c
    eigvals, eigvecs = find_H_Cardano_eigh(
        OMEGA, K, *find_epsilon_terms(OMEGA, B, N_E), np.pi / 2
    )
    root = int(np.argmin(np.abs(eigvals)))
    assert abs(eigvals[root]) < 1e-12
    e_hat = eigvecs[:, root] / eigvecs[0, root]  # u1 component 1
    i_sign = 1 if rotating_with_electrons else -1
    assert_allclose(e_hat, [1, i_sign * 1j, 0], atol=1e-12)
