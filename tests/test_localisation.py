"""Checks of the ray piece of the localisation, loc_r = g_ant^2 / g^2, and of the
group velocity g = grad_K H it uses (H the eigenvalue of D, from Cardano's formula).

Both checks need a toroidally steered beam and finite mismatch: for an unsteered
beam at small mismatch, the error from holding theta_m fixed in grad_K H vanishes.
Equation numbers refer to Hall-Chen, Parra & Hillesheim, PPCF 64 095002 (2022).
"""

from scotty.beam_me_up import beam_me_up
from scotty.init_bruv import get_parameters_for_Scotty

import numpy as np
import pytest
from scipy import constants


@pytest.fixture(
    scope="module",
    params=[(0.0, 1), (10.0, 1), (10.0, -1)],
    ids=["O-tor0", "O-tor10", "X-tor10"],
)
def steered_run(request, tmp_path_factory):
    toroidal_angle, mode_flag = request.param
    kwargs_dict = get_parameters_for_Scotty("DBS_synthetic")
    kwargs_dict["toroidal_launch_angle_Torbeam"] = toroidal_angle
    kwargs_dict["mode_flag"] = mode_flag
    kwargs_dict["figure_flag"] = False
    kwargs_dict["output_path"] = tmp_path_factory.mktemp("localisation")
    return beam_me_up(**kwargs_dict)


def test_loc_r_vacuum_limit(steered_run):
    """The beam enters the plasma where the density is zero, so the first point
    is in vacuum, where g = g_ant = 2c/omega and loc_r = 1"""
    omega = float(steered_run["inputs"]["launch_angular_frequency"])
    g_first = float(np.real(steered_run["analysis"]["g_magnitude_Cardano"].values[0]))
    loc_r_first = float(np.real(steered_run["dbs_analysis"]["loc_r"].values[0]))

    assert np.isclose(g_first * omega / (2 * constants.c), 1.0, rtol=1e-5, atol=0)
    assert np.isclose(loc_r_first, 1.0, rtol=1e-5, atol=0)


def test_g_Cardano_parallel_to_g_Booker(steered_run):
    """The Booker dispersion relation and the eigenvalue of D both vanish on
    the ray, so their K-gradients must be parallel there (eq. 46). This fails
    if theta_m is held fixed while K is varied (which makes g parallel to K),
    or if b_hat is inconsistent between the Hamiltonian and the analysis"""
    analysis = steered_run["analysis"]
    cos_angle = np.abs(
        np.sum(
            np.real(analysis["g_hat_Cardano"].values)
            * analysis["g_hat_cartesian"].values,
            axis=-1,
        )
    )
    angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    assert angle_deg.max() < 0.1


def eigenvalues_of_D(K_cart, b_cart, omega, eps_para, eps_perp, eps_g):
    """All three eigenvalues of D (eqs. 36-41), from a Hermitian eigensolver.
    Independent of Scotty's Cardano implementation"""
    K = np.linalg.norm(K_cart, axis=-1)
    sin_th = np.sum(b_cart * K_cart, axis=-1) / K
    cos_th = np.sqrt(1 - sin_th**2)
    N2 = (K * constants.c / omega) ** 2
    D = np.zeros(K.shape + (3, 3), dtype=complex)
    D[..., 0, 0] = eps_perp - N2 * sin_th**2
    D[..., 1, 1] = eps_perp - N2
    D[..., 2, 2] = eps_para - N2 * cos_th**2
    D[..., 0, 1] = -1j * eps_g
    D[..., 1, 0] = 1j * eps_g
    D[..., 0, 2] = D[..., 2, 0] = N2 * sin_th * cos_th
    return np.linalg.eigvalsh(D)


def test_g_magnitude_matches_independent_gradient(steered_run):
    """|g| = |grad_K H| with H the eigenvalue of D, recomputed independently:
    eigenvalues from a Hermitian eigensolver, derivatives in Cartesian K"""
    analysis = steered_run["analysis"]
    omega = float(steered_run["inputs"]["launch_angular_frequency"])
    K_cart = steered_run["solver_output"]["K_output_cartesian"].values
    b_cart = analysis["b_hat_cartesian"].values
    eps = [
        np.real(analysis[name].values)
        for name in ("epsilon_para", "epsilon_perp", "epsilon_g")
    ]

    root = np.argmin(np.abs(eigenvalues_of_D(K_cart, b_cart, omega, *eps)), axis=-1)
    points = np.arange(len(root))
    step = 0.1  # m^-1
    g_cart = np.zeros_like(K_cart)
    for j in range(3):
        dK = np.zeros(3)
        dK[j] = step
        H_plus = eigenvalues_of_D(K_cart + dK, b_cart, omega, *eps)[points, root]
        H_minus = eigenvalues_of_D(K_cart - dK, b_cart, omega, *eps)[points, root]
        g_cart[:, j] = (H_plus - H_minus) / (2 * step)

    np.testing.assert_allclose(
        np.real(analysis["g_magnitude_Cardano"].values),
        np.linalg.norm(g_cart, axis=-1),
        rtol=1e-3,
    )
