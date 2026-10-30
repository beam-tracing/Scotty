"""Physics invariants of a full Scotty run, checked on cases where errors can show.

Golden values only record what the code did when they were written. These tests
instead check properties any correct result must have, on a toroidally steered
beam and one with large mismatch (most analysis errors vanish for an unsteered
beam at zero mismatch, which is what the golden tests use).

Equation numbers refer to Hall-Chen, Parra & Hillesheim, PPCF 64 095002 (2022).
"""

from scotty.beam_me_up import beam_me_up
from scotty.init_bruv import get_parameters_for_Scotty

import numpy as np
import pytest
from scipy import constants

# (toroidal launch angle in degrees, mode_flag). DBS_synthetic at these angles has
# theta_m at the cut-off of about +8 deg, -17 deg, -22 deg (X-mode) and -44 deg
CASES = {
    "O-tor0": (0.0, 1),
    "O-tor8": (8.0, 1),
    "X-tor8": (8.0, -1),
    "O-tor20": (20.0, 1),
}


def run_synthetic(output_path, toroidal_angle, mode_flag, **kwargs):
    output_path.mkdir(parents=True, exist_ok=True)
    kwargs_dict = get_parameters_for_Scotty("DBS_synthetic")
    kwargs_dict["find_B_method"] = "unit-tests"
    kwargs_dict["toroidal_launch_angle_Torbeam"] = toroidal_angle
    kwargs_dict["mode_flag"] = mode_flag
    kwargs_dict["figure_flag"] = False
    kwargs_dict["output_path"] = output_path
    kwargs_dict.update(kwargs)
    return beam_me_up(**kwargs_dict)


@pytest.fixture(scope="module", params=list(CASES.values()), ids=list(CASES.keys()))
def case(request, tmp_path_factory):
    toroidal_angle, mode_flag = request.param
    dt = run_synthetic(tmp_path_factory.mktemp("invariants"), toroidal_angle, mode_flag)
    return dict(dt=dt, toroidal_angle=toroidal_angle, mode_flag=mode_flag)


def test_dispersion_relation_satisfied_on_ray(case):
    """The ray is traced on H = 0 (eq. 47), so the Booker H must stay near zero"""
    H = case["dt"]["analysis"]["H_Booker"].values
    assert np.max(np.abs(H)) < 1e-3


def test_polarisation_matches_mode(case):
    """At the cut-off, O-mode is polarised mostly along b, X-mode mostly across it.
    e_hat is in Scotty's (e_1, e_2, e_b) basis"""
    analysis = case["dt"]["analysis"]
    e_hat = np.abs(analysis["e_hat"].values[int(analysis["cutoff_index"])])
    e_b = e_hat[2] / np.linalg.norm(e_hat)
    if case["mode_flag"] == 1:
        assert e_b > 0.7
    else:
        assert e_b < 0.2


def test_analysis_outputs_finite_and_in_range(case):
    """Every localisation piece is finite along the ray, loc_m is a Gaussian
    attenuation in (0, 1], and the mismatch tolerance is positive"""
    analysis = case["dt"]["analysis"]
    for name in ("loc_p", "loc_b", "loc_r", "loc_m", "loc_s", "theta_m", "k_perp_1_bs"):
        values = np.asarray(analysis[name].values)
        assert np.all(np.isfinite(values)), name
    loc_m = np.real(analysis["loc_m"].values)
    assert np.all((loc_m > 0) & (loc_m <= 1))
    assert np.all(np.real(analysis["delta_theta_m"].values) > 0)
    assert np.all(np.real(analysis["loc_r"].values) > 0)


def eigenvalue_of_D(K_cart, b_cart, omega, eps_para, eps_perp, eps_g):
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


def test_g_magnitude_matches_independent_cartesian_gradient(case):
    """|g| = |grad_K H| with H the eigenvalue of D, recomputed independently:
    eigenvalues from a Hermitian eigensolver, derivatives in Cartesian K
    (no metric factors). Catches unit errors in the toroidal component and
    any K-dependence (such as theta_m) held fixed by mistake"""
    dt = case["dt"]
    analysis, solver = dt["analysis"], dt["solver_output"]
    omega = float(dt["inputs"]["launch_angular_frequency"])
    q_R, q_zeta = solver["q_R"].values, solver["q_zeta"].values
    K_zeta = float(analysis["K_zeta_initial"])
    R_hat = np.stack([np.cos(q_zeta), np.sin(q_zeta), 0 * q_zeta], axis=-1)
    zeta_hat = np.stack([-np.sin(q_zeta), np.cos(q_zeta), 0 * q_zeta], axis=-1)
    Z_hat = np.stack([0 * q_zeta, 0 * q_zeta, 1 + 0 * q_zeta], axis=-1)

    def to_cartesian(v):
        return v[:, :1] * R_hat + v[:, 1:2] * zeta_hat + v[:, 2:] * Z_hat

    K_cyl = np.stack(
        [analysis["K_R"].values, K_zeta / q_R, analysis["K_Z"].values], axis=-1
    )
    K_cart = to_cartesian(K_cyl)
    b_cart = to_cartesian(np.real(analysis["b_hat"].values))
    eps = [
        np.real(analysis[name].values)
        for name in ("epsilon_para", "epsilon_perp", "epsilon_g")
    ]

    # The eigenvalue that is zero on the ray
    root = np.argmin(np.abs(eigenvalue_of_D(K_cart, b_cart, omega, *eps)), axis=-1)
    points = np.arange(len(root))

    step = 0.1  # m^-1
    g_cart = np.zeros_like(K_cart)
    for j in range(3):
        dK = np.zeros(3)
        dK[j] = step
        H_plus = eigenvalue_of_D(K_cart + dK, b_cart, omega, *eps)[points, root]
        H_minus = eigenvalue_of_D(K_cart - dK, b_cart, omega, *eps)[points, root]
        g_cart[:, j] = (H_plus - H_minus) / (2 * step)

    g_magnitude = np.real(analysis["g_magnitude_Cardano"].values)
    np.testing.assert_allclose(g_magnitude, np.linalg.norm(g_cart, axis=-1), rtol=1e-3)


@pytest.mark.xfail(
    strict=True,
    reason="Launch zeta is used as a Cartesian Y coordinate (launch.py, find_entry_point), "
    "so a launch at zeta != 0 does not give the rotated beam",
)
def test_toroidal_rotation_invariance(tmp_path):
    """The field is axisymmetric, so launching from zeta = 0.5 rad must give the
    same beam rotated by 0.5 rad"""
    params = get_parameters_for_Scotty("DBS_synthetic")
    shifted = np.array(params["launch_position"], dtype=float)
    shifted[1] += 0.5
    base = run_synthetic(tmp_path / "base", 8.0, 1)
    rotated = run_synthetic(tmp_path / "rotated", 8.0, 1, launch_position=shifted)

    for name in ("q_R", "q_Z", "K_R", "K_Z"):
        np.testing.assert_allclose(
            rotated["solver_output"][name],
            base["solver_output"][name],
            rtol=1e-6,
            atol=1e-9,
            err_msg=name,
        )
    np.testing.assert_allclose(
        rotated["solver_output"]["q_zeta"],
        base["solver_output"]["q_zeta"] + 0.5,
        rtol=0,
        atol=1e-6,
    )


def test_cutoff_quantities_converged_in_solver_tolerance(tmp_path):
    """Quantities at the cut-off do not depend on the solver tolerance (default
    rtol = 1e-3, atol = 1e-6) beyond a small, stated amount"""
    default = run_synthetic(tmp_path / "default", 8.0, 1)
    tight = run_synthetic(tmp_path / "tight", 8.0, 1, rtol=1e-6, atol=1e-9)

    def at_cutoff(dt):
        a = dt["analysis"]
        i = int(a["cutoff_index"])
        loc = np.real(a["loc_p"] * a["loc_b"] * a["loc_r"] * a["loc_m"]).values[i]
        return dict(
            q_R=float(dt["solver_output"]["q_R"][i]),
            q_Z=float(dt["solver_output"]["q_Z"][i]),
            theta_m=float(a["theta_m"][i]),
            k_perp_1_bs=float(np.real(a["k_perp_1_bs"][i])),
            loc=float(loc),
        )

    d, t = at_cutoff(default), at_cutoff(tight)
    np.testing.assert_allclose(d["q_R"], t["q_R"], rtol=0, atol=1e-3)  # metres
    np.testing.assert_allclose(d["q_Z"], t["q_Z"], rtol=0, atol=1e-3)
    np.testing.assert_allclose(d["theta_m"], t["theta_m"], rtol=0, atol=np.radians(0.1))
    np.testing.assert_allclose(d["k_perp_1_bs"], t["k_perp_1_bs"], rtol=1e-2)
    np.testing.assert_allclose(d["loc"], t["loc"], rtol=5e-2)
