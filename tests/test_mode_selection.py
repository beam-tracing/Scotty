"""Telling the O- and X-modes apart, including at zero density.

At zero density both Booker roots have N = 1 and D is degenerate in the plane
transverse to K, so any transverse polarisation is allowed and K alone cannot
tell the modes apart. Their polarisations still tend to distinct limits as the
density goes to zero (degenerate perturbation theory): the O-mode is the root
more polarised along b. These tests check those limits, that "O" and "X" pick
the right root for vacuum and plasma launches, and that the resonance event is
told which root is the X-mode.
"""

from scotty.beam_me_up import beam_me_up
from scotty.fun_general import freq_GHz_to_angular_frequency
from scotty.init_bruv import get_parameters_for_Scotty
from scotty.launch import (
    POLARISATION_PROBE_DENSITY,
    _mode_polarisations,
    _O_mode_flag,
)

import importlib
import numpy as np
from numpy.testing import assert_allclose
import pytest

# The module, not the function: scotty/__init__.py re-exports `beam_me_up`
# under the module's name
beam_me_up_module = importlib.import_module("scotty.beam_me_up")

OMEGA = freq_GHz_to_angular_frequency(55.0)
B = 0.75  # Tesla, as at DBS_synthetic's plasma entry


def e_b(polarisations):
    return {m: abs(e_hat[-1]) for m, e_hat in polarisations.items()}


##################################################
# The polarisations' zero-density limits
##################################################


@pytest.mark.parametrize("theta_deg", [0.0, 10.0, 30.0])
def test_polarisations_tend_to_distinct_limits(theta_deg):
    theta_m = np.deg2rad(theta_deg)
    at = {n: e_b(_mode_polarisations(OMEGA, B, theta_m, n)) for n in (1e-5, 1e-9)}

    for m in (1, -1):
        assert np.isclose(at[1e-5][m], at[1e-9][m], atol=1e-6)
    assert abs(at[1e-9][1] - at[1e-9][-1]) > 0.1


@pytest.mark.parametrize("theta_deg", [0.0, 10.0, 30.0])
def test_zero_density_uses_the_limit(theta_deg):
    """At exactly zero density the probe density is used, which gives the
    limit rather than an arbitrary degenerate eigenvector"""
    theta_m = np.deg2rad(theta_deg)
    zero = e_b(_mode_polarisations(OMEGA, B, theta_m, 0.0))
    small = e_b(_mode_polarisations(OMEGA, B, theta_m, 1e-9))

    assert POLARISATION_PROBE_DENSITY > 1e-9
    for m in (1, -1):
        assert np.isclose(zero[m], small[m], atol=1e-6)


def test_perpendicular_O_mode_is_along_b():
    """For K perpendicular to B, the O-mode's electric field is along B"""
    polarisations = _mode_polarisations(OMEGA, B, 0.0, 0.0)
    m_O = _O_mode_flag(polarisations)

    assert np.isclose(abs(polarisations[m_O][-1]), 1.0, atol=1e-3)
    assert abs(polarisations[-m_O][-1]) < 0.1


def test_modes_along_B_cannot_be_told_apart():
    """Along B both modes are circularly polarised transverse to b"""
    polarisations = _mode_polarisations(OMEGA, B, np.deg2rad(89.99), 0.0)

    with pytest.raises(ValueError, match="too alike"):
        _O_mode_flag(polarisations)


##################################################
# Named modes in beam_me_up
##################################################


def synthetic(tmp_path, **changes):
    kwargs = get_parameters_for_Scotty("DBS_synthetic")
    kwargs.update(figure_flag=False, len_tau=20, output_path=tmp_path)
    kwargs.update(changes)
    return kwargs


def inside_e_b(dt):
    """|e_b| a quarter of the way along the beam, inside the plasma"""
    e_hat = dt["analysis"]["e_hat"].values
    return abs(e_hat[len(e_hat) // 4, -1])


@pytest.fixture(scope="module", params=[0.0, 10.0], ids=["tor0", "tor10"])
def vacuum_runs(request, tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("modes")
    return {
        mode_flag: beam_me_up(
            **synthetic(
                tmp_path,
                mode_flag=mode_flag,
                toroidal_launch_angle_Torbeam=request.param,
            )
        )
        for mode_flag in (1, -1, "O", "X")
    }


def test_vacuum_named_modes_trace_the_right_root(vacuum_runs):
    """ "O" traces the root that is O-polarised inside the plasma, and "X" the
    other one (before this fix, both traced the X-mode)"""
    O_root = 1 if inside_e_b(vacuum_runs[1]) > inside_e_b(vacuum_runs[-1]) else -1

    assert int(vacuum_runs["O"]["solver_output"]["mode_flag_initial"]) == O_root
    assert int(vacuum_runs["X"]["solver_output"]["mode_flag_initial"]) == -O_root


def test_vacuum_named_mode_is_the_numeric_one(vacuum_runs):
    """ "O" and "X" give the same beam as their numeric mode_flag"""
    for named in ("O", "X"):
        numeric = int(vacuum_runs[named]["solver_output"]["mode_flag_initial"])
        assert_allclose(
            vacuum_runs[named]["solver_output"]["q_output_cylindrical"],
            vacuum_runs[numeric]["solver_output"]["q_output_cylindrical"],
        )


def test_e_hat_initial_is_the_zero_density_limit(vacuum_runs):
    """The beam enters where the density is zero; e_hat_initial is the mode's
    zero-density polarisation, not an arbitrary degenerate vector"""
    for mode_flag in (1, -1):
        e_hat = vacuum_runs[mode_flag]["solver_output"]["e_hat_initial"].values
        assert np.isclose(np.linalg.norm(e_hat), 1.0)
    e_b_O = abs(vacuum_runs[1]["solver_output"]["e_hat_initial"].values[-1])
    e_b_X = abs(vacuum_runs[-1]["solver_output"]["e_hat_initial"].values[-1])
    assert e_b_O > e_b_X + 0.1


##################################################
# The resonance event is told which root is the X-mode
##################################################


@pytest.mark.parametrize(
    ("mode_flag", "X_mode"), [(1, False), (-1, True), ("O", False), ("X", True)]
)
def test_beam_me_up_passes_X_mode_to_the_ray_solver(
    tmp_path, monkeypatch, mode_flag, X_mode
):
    """For DBS_synthetic the X-mode is mode_flag = -1, given as -1 or "X\" """
    seen = {}
    ray_tracing = beam_me_up_module.ray_tracing

    def spy(**kwargs):
        seen["X_mode"] = kwargs["X_mode"]
        return ray_tracing(**kwargs)

    monkeypatch.setattr(beam_me_up_module, "ray_tracing", spy)
    beam_me_up(**synthetic(tmp_path, mode_flag=mode_flag, len_tau=10))

    assert seen["X_mode"] is X_mode


@pytest.mark.parametrize("mode_flag", [1, -1])
def test_numeric_mode_flag_needs_no_label(tmp_path, mode_flag):
    """Far enough from perpendicular, the polarisations are too alike to label,
    but 1 or -1 still says which root to trace."""
    dt = beam_me_up(
        **synthetic(tmp_path, mode_flag=mode_flag, toroidal_launch_angle_Torbeam=45.0)
    )
    assert int(dt["solver_output"]["mode_flag_initial"]) == mode_flag


def test_named_mode_raises_when_modes_cannot_be_labelled(tmp_path):
    with pytest.raises(ValueError, match="too alike"):
        beam_me_up(
            **synthetic(tmp_path, mode_flag="O", toroidal_launch_angle_Torbeam=45.0)
        )


@pytest.mark.parametrize("poloidal_flux_enter", [0.999, 0.99])
@pytest.mark.parametrize("mode_flag", [1, -1])
def test_e_hat_initial_is_the_traced_roots_at_low_density(tmp_path, poloidal_flux_enter, mode_flag):
    """Each root's own polarisation, as on the ray's first point, not the same
    vector for both."""
    dt = beam_me_up(
        **synthetic(tmp_path, mode_flag=mode_flag, poloidal_flux_enter=poloidal_flux_enter)
    )
    assert_allclose(
        np.abs(dt["solver_output"]["e_hat_initial"].values),
        np.abs(dt["analysis"]["e_hat"].values[0]),
        atol=0.01,
    )
