import pathlib

import numpy as np
from numpy.testing import assert_allclose

from scotty.beam_me_up import beam_me_up
from scotty.compat import V3_ONLY_ARGUMENTS, from_v3_kwargs
from scotty.fun_general import find_K_labframe_cart_to_cyl, find_q_labframe_cyl_to_cart
from scotty.init_bruv import get_parameters_for_Scotty


def v3_style(kwargs: dict) -> dict:
    """A v4 parameters dict rewritten as a v3 script would have passed it"""
    kwargs = dict(kwargs)
    for name in ("geometry", "launch_flag", "boundary_flag"):
        kwargs.pop(name)
    kwargs.update(vacuumLaunch_flag=True, vacuum_propagation_flag=True, Psi_BC_flag="continuous")
    return kwargs


def test_launch_flags_converted():
    kw = from_v3_kwargs({"vacuumLaunch_flag": True, "Psi_BC_flag": "continuous"})
    assert kw["launch_flag"] == "vacuum"
    assert kw["boundary_flag"] == "continuous"
    assert kw["geometry"] == "cylindrical"
    assert not set(V3_ONLY_ARGUMENTS) & set(kw)

    kw = from_v3_kwargs({"vacuumLaunch_flag": False, "Psi_BC_flag": True})
    assert kw["launch_flag"] == "plasma"
    assert kw["boundary_flag"] == "continuous"  # v3 treated True as "continuous"


def test_unset_arguments_get_v3_defaults():
    kw = from_v3_kwargs({"poloidal_flux_enter": 1.2})
    assert kw["launch_flag"] == "vacuum"
    assert kw["boundary_flag"] is None
    assert kw["interp_smoothing"] == 0
    assert kw["find_B_method"] == "torbeam"
    assert kw["detailed_analysis_flag"] is True
    # v3's 1.01, raised to poloidal_flux_enter as v4 requires
    assert kw["poloidal_flux_zero_density"] == 1.2
    assert kw["poloidal_flux_zero_temperature"] == 1.2


def test_explicit_arguments_kept_and_idempotent():
    given = {"geometry": "cylindrical", "launch_flag": "plasma", "boundary_flag": "discontinuous",
             "interp_smoothing": 3, "poloidal_flux_zero_density": 1.1, "Psi_BC_flag": None}
    kw = from_v3_kwargs(given)
    for name in ("launch_flag", "boundary_flag", "interp_smoothing", "poloidal_flux_zero_density"):
        assert kw[name] == given[name]
    assert from_v3_kwargs(kw) == kw


def test_output_path_and_quick_run():
    kw = from_v3_kwargs({"output_path": "runs/case1/", "quick_run": True, "resonance_flag": False})
    assert kw["output_path"] == pathlib.Path("runs/case1")
    assert kw["ray_tracing_flag"] is True
    assert "resonance_flag" not in kw


def test_plasma_launch_K_cylindrical_to_cartesian():
    launch_position = np.array([1.8, 0.3, 0.1])
    K_cyl = np.array([-900.0, 50.0, 20.0])
    kw = from_v3_kwargs({"vacuumLaunch_flag": False, "launch_position": launch_position,
                         "plasmaLaunch_K": K_cyl, "plasmaLaunch_Psi_3D_lab_Cartesian": np.eye(3)})
    # The inverse takes the position in Cartesian
    q_cart = find_q_labframe_cyl_to_cart(launch_position)
    K_back = find_K_labframe_cart_to_cyl(kw["plasmaLaunch_K_cartesian"], q_cart)
    assert_allclose(K_back, K_cyl)
    assert_allclose(kw["plasmaLaunch_Psi_3D_lab_cartesian"], np.eye(3))


def test_v3_script_gives_same_result_as_v4(tmp_path):
    v4 = get_parameters_for_Scotty("DBS_synthetic")
    v4.update(len_tau=52, interp_smoothing=0)
    v3 = from_v3_kwargs(v3_style(v4))

    out_v4 = beam_me_up(**v4, output_path=tmp_path, output_filename_suffix="_v4")
    out_v3 = beam_me_up(**v3, output_path=tmp_path, output_filename_suffix="_v3")

    for group, name in (("solver_output", "q_R_output"), ("solver_output", "K_R_output"),
                        ("analysis", "theta_m")):
        assert_allclose(out_v3[group][name], out_v4[group][name])


def test_v3_arguments_point_to_the_converter(caplog):
    from scotty.checks import Parameters

    params = Parameters.__new__(Parameters)
    params._unpack_unused_kwargs(extras={"Psi_BC_flag": "continuous", "vacuumLaunch_flag": True})
    assert "scotty.compat.from_v3_kwargs" in caplog.text
    assert "Psi_BC_flag" in caplog.text and "vacuumLaunch_flag" in caplog.text
