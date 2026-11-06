"""Invariants of the analysis on a toroidally steered beam.

The golden tests launch at a toroidal angle of zero, where the beam stays near
zeta = 0 and errors in the cylindrical-to-Cartesian conversion of the field
vanish. These cases steer the beam toroidally so that zeta changes along the ray.
"""

from scotty.analysis import find_mode_index_Cardano
from scotty.checks import check_H_output
from scotty.beam_me_up import beam_me_up
from scotty.init_bruv import get_parameters_for_Scotty

import logging

import numpy as np
import pytest
from scipy.interpolate import CubicSpline


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
    kwargs_dict["output_path"] = tmp_path_factory.mktemp("analysis_invariants")
    return beam_me_up(**kwargs_dict)


def test_b_hat_matches_field_components(steered_run):
    """b_hat_cartesian is the unit vector of the stored (B_X, B_Y, B_Z)"""
    analysis = steered_run["beam"]
    B = np.stack(
        [analysis["B_X"].values, analysis["B_Y"].values, analysis["B_Z"].values],
        axis=-1,
    )
    np.testing.assert_allclose(
        analysis["b_hat_cartesian"].values,
        B / np.linalg.norm(B, axis=-1, keepdims=True),
        rtol=0,
        atol=1e-10,
    )


def test_polarisation_matches_mode(steered_run, request):
    """At the cut-off, O-mode is polarised mostly along b and X-mode mostly
    across it. e_hat is an eigenvector of D, so it is in the (e_1, e_2, b)
    basis of eqs. 36-41 (despite its "col_cart" label) and the b-component
    is its third entry. Fails if e_hat is taken from the wrong eigenvector
    (the eigh ordering is not the Cardano ordering)"""
    analysis = steered_run["beam"]
    i = int(analysis["cutoff_index"])
    e_hat = analysis["e_hat"].values[i]
    e_b = np.abs(e_hat[2]) / np.linalg.norm(e_hat)

    if "X-" in request.node.callspec.id:
        assert e_b < 0.2
    else:
        assert e_b > 0.7


def test_eigenvalue_of_D_consistent_with_Booker(steered_run):
    """The Booker H and the eigenvalue of D (Cardano) vanish on the same
    surface, so on the ray the Cardano H can be no larger than the solver's
    drift off that surface. An inconsistent b_hat between the Hamiltonian and
    the analysis breaks this"""
    H_Cardano = np.real(steered_run["checks"]["H_Cardano_solution"].values)
    H_Booker = np.real(steered_run["checks"]["H_Booker"].values)

    assert np.max(np.abs(H_Cardano)) < 2 * np.max(np.abs(H_Booker)) + 1e-6


def test_dispersion_relation_satisfied_on_ray(steered_run):
    """The ray is traced on H = 0, so the Booker H must stay near zero, to within
    the solver's step error. At rtol 1e-3, a steered X-mode ray drifts
    to |H| ~ 4e-3 in its last few points (step error on K ~ 1000 m^-1), so the
    bound scales with rtol rather than being fixed"""
    rtol = float(steered_run["inputs"]["rtol"])
    H_Booker = np.real(steered_run["checks"]["H_Booker"].values)

    assert np.max(np.abs(H_Booker)) < 10 * rtol


def test_dispersion_drift_is_step_error(tmp_path):
    """The drift off H = 0 of the steered X-mode ray is solver step error, not an
    error in H: tightening the tolerance from 1e-3 to 1e-5 shrinks it by far more
    than 100 (measured: 4e-3 to 5e-7)"""

    def max_drift(rtol, atol, folder):
        kwargs_dict = get_parameters_for_Scotty("DBS_synthetic")
        kwargs_dict.update(
            toroidal_launch_angle_Torbeam=10.0,
            mode_flag=-1,
            figure_flag=False,
            output_path=tmp_path / folder,
            rtol=rtol,
            atol=atol,
        )
        (tmp_path / folder).mkdir()
        output = beam_me_up(**kwargs_dict)
        return np.max(np.abs(np.real(output["checks"]["H_Booker"].values)))

    loose = max_drift(1e-3, 1e-6, "loose")
    tight = max_drift(1e-5, 1e-8, "tight")

    assert tight < 1e-5
    assert tight < loose / 100


def test_mode_index_Cardano_consistent(caplog):
    abs_H = np.array([np.ones(10), np.zeros(10), np.ones(10)])
    with caplog.at_level(logging.WARNING):
        assert find_mode_index_Cardano(abs_H) == 1
    assert caplog.text == ""


def test_mode_index_Cardano_some_points_near_another_root(caplog):
    """As near the plasma edge, where two roots are close together: points nearer
    the other one, from step error, don't stop the run or change the root"""
    abs_H = np.array([np.ones(100), np.full(100, 1e-4), np.full(100, 0.1)])
    abs_H[2, -10:] = 1e-5
    with caplog.at_level(logging.WARNING):
        assert find_mode_index_Cardano(abs_H) == 1
    assert "10 of 100 points" in caplog.text
    assert "up to 1.00e-04" in caplog.text
    assert "`rtol`" in caplog.text


def test_mode_index_Cardano_long_low_density_stretch():
    """Most of the ray at low density, where two roots are both near zero and
    which is the smaller is down to rounding: the root isn't chosen by a vote"""
    abs_H = np.array([np.ones(100), np.full(100, 1e-4), np.full(100, 1e-4)])
    abs_H[2, :70] = 0.9e-4
    abs_H[2, 70:] = 0.5
    assert find_mode_index_Cardano(abs_H) == 1


def test_mode_index_Cardano_warns_if_the_ray_moves_to_another_root(caplog):
    """The labels are the branches of the Cardano formula. Moving between them isn't
    fatal here: `check_H_output` judges the size of |H| on the root chosen"""
    abs_H = np.array(
        [
            np.ones(6),
            [1e-4, 1e-4, 1e-4, 1e-4, 0.9, 0.9],
            [0.9, 0.9, 0.9, 0.9, 1e-4, 1e-4],
        ]
    )
    with caplog.at_level(logging.WARNING):
        assert find_mode_index_Cardano(abs_H) == 1
    assert "2 of 6 points" in caplog.text
    assert "up to 9.00e-01" in caplog.text


def test_mode_index_Cardano_nan_in_another_root_is_ignored():
    abs_H = np.array([np.ones(6), np.full(6, 1e-4), np.full(6, 0.1)])
    abs_H[2, 1] = np.nan
    assert find_mode_index_Cardano(abs_H) == 1


def test_mode_index_Cardano_does_not_favour_a_root_with_many_nans():
    abs_H = np.array([np.ones(6), np.full(6, 1e-4), np.full(6, np.nan)])
    abs_H[2, 0] = 1e-6
    assert find_mode_index_Cardano(abs_H) == 1


def test_mode_index_Cardano_nan_in_the_ray_root_leaves_it_to_the_check():
    """A NaN on the ray's own root: another root is chosen, whose large |H| is then
    for `check_H_output` to judge"""
    abs_H = np.array([np.ones(6), np.full(6, 1e-4), np.full(6, 0.1)])
    abs_H[1, 1] = np.nan
    assert find_mode_index_Cardano(abs_H) == 2


def test_mode_index_Cardano_nan_in_every_root_at_one_point():
    abs_H = np.array([np.ones(6), np.full(6, 1e-4), np.full(6, 0.1)])
    abs_H[:, 1] = np.nan
    assert find_mode_index_Cardano(abs_H) == 1


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_check_H_output_sees_either_sign(sign):
    H = np.zeros(10)
    H[5] = sign * 2.0
    with pytest.raises(ValueError, match="unphysically large"):
        check_H_output(H)


@pytest.fixture(
    scope="module",
    params=[(0.0, 1), (10.0, 1), (10.0, -1)],
    ids=["O-tor0", "O-tor10", "X-tor10"],
)
def tight_run(request, tmp_path_factory):
    toroidal_angle, mode_flag = request.param
    kwargs_dict = get_parameters_for_Scotty("DBS_synthetic")
    kwargs_dict["toroidal_launch_angle_Torbeam"] = toroidal_angle
    kwargs_dict["mode_flag"] = mode_flag
    kwargs_dict["figure_flag"] = False
    kwargs_dict["rtol"] = 1e-5
    kwargs_dict["atol"] = 1e-8
    kwargs_dict["output_path"] = tmp_path_factory.mktemp("analysis_invariants")
    return beam_me_up(**kwargs_dict)


def test_g_matches_dq_dtau(tight_run):
    """The ray equation dq/dtau = g, comparing the path the
    solver integrated with the g the analysis recalculates from H. This tests
    size and sign as well as direction, and fails if the solver and the analysis
    disagree on H or on tau. The difference falls as rtol is tightened, so the
    run needs a tighter rtol than the default for the test to be sharp"""
    solver_output = tight_run["solver_output"]
    tau = solver_output["tau"].values
    dq_dtau = np.stack(
        [
            CubicSpline(tau, solver_output[name].values)(tau, 1)
            for name in ("q_X_output", "q_Y_output", "q_Z_output")
        ],
        axis=-1,
    )
    g = np.real(tight_run["analysis"]["g_vector_cartesian"].values)

    relative_difference = np.linalg.norm(dq_dtau - g, axis=-1) / np.linalg.norm(
        g, axis=-1
    )
    assert relative_difference.max() < 1e-4
