"""Tests for known bugs that are not fixed yet.

Each test states the correct behaviour and is marked ``xfail(strict=True)``, so
it fails today, and the suite fails as soon as the bug is fixed (an unexpected
pass), as a reminder to remove the marker. The reason names the code at fault.
"""

import importlib
import pkgutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

import scotty
from scotty.beam_me_up import beam_me_up
from scotty.check_output import check_H_output
from scotty.fun_general import find_q_lab_Cartesian
from scotty.init_bruv import beam_settings, get_parameters_for_Scotty
from scotty.lensalot import ABCD_Lens
from scotty.profile_fit import QuadraticFit


def known_bug(reason):
    return pytest.mark.xfail(strict=True, reason=reason)


@known_bug("find_q_lab_Cartesian uses q_zeta where it should use q_R (fun_general.py)")
def test_q_lab_cartesian_off_the_xz_plane():
    q_lab = np.array([[1.0], [np.pi / 2], [0.3]])
    np.testing.assert_allclose(find_q_lab_Cartesian(q_lab)[:, 0], [0.0, 1.0, 0.3], atol=1e-12)


@known_bug("QuadraticFit divides by psi_0 instead of psi_0**2, so it reaches zero at sqrt(psi_0) (profile_fit.py)")
def test_quadratic_fit_goes_to_zero_at_psi_0():
    fit = QuadraticFit(poloidal_flux_zero_profile=0.8, ne_0=2.0)
    assert np.isclose(fit(0.8 - 1e-9), 0.0, atol=1e-6)


@known_bug("ABCD_Lens uses Psi_w_in[0, 0] for both diagonal elements (lensalot.py)")
def test_abcd_identity_lens_returns_input():
    lens = ABCD_Lens("identity", A=1.0, B=0.0, C=0.0, D=1.0)
    Psi_w_in = np.diag([100.0 + 50.0j, 300.0 + 20.0j])
    np.testing.assert_allclose(lens.output_beam(Psi_w_in, 55.0), Psi_w_in, rtol=1e-12)


@pytest.mark.parametrize("H_value", [5.0, -5.0])
@known_bug("check_H_output: the raise branch is unreachable after the warning, and max(H) ignores negative H (check_output.py)")
def test_check_H_output_raises_on_large_H(H_value):
    with pytest.raises(ValueError):
        check_H_output(np.array([0.0, H_value, 0.0]))


@known_bug("horn_and_lens tests '> 52.5' and '< 52.5', so 52.5 GHz leaves the horn undefined (init_bruv.py)")
def test_horn_and_lens_at_band_edge():
    width, curvature = beam_settings("DBS_NSTX_MAST", 52.5, "horn_and_lens")
    assert width > 0


@pytest.mark.parametrize("method", ["estimate_var_w0", "estimate_fix_w0"])
@known_bug("SWIP presets call propagate_circular_beam(distance, K0, w0, f) but the signature is (width, curvature, distance, f) (init_bruv.py)")
def test_swip_launch_width_is_physical(method):
    width, curvature = beam_settings("DBS_SWIP_MAST-U", 52.0, method)
    assert 0 < width < 0.2


@known_bug("event_reach_K_min uses q_R * dH_dKR where d|K|/dtau needs (K_zeta**2 / q_R**3) * dH_dKR (ray_solver.py)")
def test_quick_run_cutoff_matches_full_run_for_steered_beam(tmp_path):
    def run(quick, folder):
        path = tmp_path / folder
        path.mkdir()
        kwargs_dict = get_parameters_for_Scotty("DBS_synthetic")
        kwargs_dict.update(find_B_method="unit-tests", figure_flag=False, output_path=path,
                           toroidal_launch_angle_Torbeam=20.0, quick_run=quick)
        return beam_me_up(**kwargs_dict)

    full = run(False, "full")
    quick = run(True, "quick")
    i = int(full["analysis"]["cutoff_index"])
    assert abs(quick.q_R - float(full["solver_output"]["q_R"][i])) < 1e-3  # metres
    assert abs(quick.q_Z - float(full["solver_output"]["q_Z"][i])) < 1e-3


@known_bug("beam_me_up does not create output_path, and only fails when writing the output at the end")
def test_beam_me_up_creates_output_path():
    with tempfile.TemporaryDirectory() as tmp:
        kwargs_dict = get_parameters_for_Scotty("DBS_synthetic")
        kwargs_dict.update(find_B_method="unit-tests", figure_flag=False, len_tau=10,
                           output_path=Path(tmp) / "does" / "not" / "exist")
        beam_me_up(**kwargs_dict)


BROKEN_IMPORTS = {
    "scotty.check_ModeConversion": "imports contract_special, which no longer exists in fun_general",
    "scotty.compare_plot": "imports contract_special, which no longer exists in fun_general",
}


@pytest.mark.parametrize(
    "module",
    [
        pytest.param(m.name, marks=known_bug(BROKEN_IMPORTS[m.name]) if m.name in BROKEN_IMPORTS else ())
        for m in pkgutil.iter_modules(scotty.__path__, "scotty.")
    ],
)
def test_every_module_imports(module):
    importlib.import_module(module)
