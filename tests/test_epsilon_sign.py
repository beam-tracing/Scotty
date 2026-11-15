"""The sign of epsilon_12 (`epsilon_g`).

With fields ~ exp(-i Omega t) and b_hat = B / |B|, the electrons (charge -e)
gyrate in the positive sense about b_hat, so of the two circularly polarised
waves along B, the one rotating with them, u1 + i u2, is the one that
resonates at Omega -> Omega_ce: N^2 = 1 - Omega_pe^2 / (Omega (Omega -
Omega_ce)). Scotty keeps the charge unsigned (Omega_ce = e B / m_e > 0), so
epsilon_12 = -Omega_pe^2 Omega_ce / (Omega (Omega^2 - Omega_ce^2)), negative
for Omega > Omega_ce. Hall-Chen, Parra & Hillesheim (2022) eqs. (2) and (35)
have the opposite sign.
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
B = 1.0  # T, so Omega_ce / Omega = 0.51
N_E = 0.5  # 10^19 m^-3, so Omega_pe^2 / Omega^2 = 0.13


def omega_pe_and_omega_ce():
    Omega_pe = find_normalised_plasma_ang_freq(OMEGA, N_E) * OMEGA
    Omega_ce = find_normalised_gyro_ang_freq(OMEGA, B) * OMEGA
    return Omega_pe, Omega_ce


def test_epsilon_12_is_negative_with_the_charge_unsigned():
    """The formula, in both places it is computed (a regression check; the
    physics is in the test below)"""
    Omega_pe, Omega_ce = omega_pe_and_omega_ce()
    assert 0 < Omega_ce < OMEGA
    assert Omega_pe**2 < OMEGA * (OMEGA - Omega_ce)  # both waves propagate
    expected = -(Omega_pe**2) * Omega_ce / (OMEGA * (OMEGA**2 - Omega_ce**2))
    assert_allclose(DielectricTensor(OMEGA, B, N_E).e_12, expected, rtol=1e-12)
    assert_allclose(find_epsilon_terms(OMEGA, B, N_E)[2], expected, rtol=1e-12)


def N_sq_along_B(Omega_pe, Omega_ce, rotating_with_electrons):
    """N^2 of the wave along B rotating with (or against) the electrons"""
    sign = -1 if rotating_with_electrons else 1
    return 1 - Omega_pe**2 / (OMEGA * (OMEGA + sign * Omega_ce))


@pytest.mark.parametrize("rotating_with_electrons", [True, False])
def test_circular_polarisation_along_B(rotating_with_electrons):
    """K along b_hat (theta_m = 90 degrees): the root of D at N^2 = 1 -
    Omega_pe^2 / (Omega (Omega - Omega_ce)) has polarisation u1 + i u2,
    rotating with the electrons, and the root at N^2 = 1 - Omega_pe^2 /
    (Omega (Omega + Omega_ce)) has u1 - i u2"""
    N_sq = N_sq_along_B(*omega_pe_and_omega_ce(), rotating_with_electrons)
    K = np.sqrt(N_sq) * OMEGA / constants.c
    eigvals, eigvecs = find_H_Cardano_eigh(
        OMEGA, K, *find_epsilon_terms(OMEGA, B, N_E), np.pi / 2
    )
    root = int(np.argmin(np.abs(eigvals)))
    assert abs(eigvals[root]) < 1e-12
    e_hat = eigvecs[:, root] / eigvecs[0, root]  # u1 component 1
    i_sign = 1 if rotating_with_electrons else -1
    assert_allclose(e_hat, [1, i_sign * 1j, 0], atol=1e-12)
