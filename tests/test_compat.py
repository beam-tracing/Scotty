import pathlib

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from scotty.beam_me_up import beam_me_up
from scotty.compat import (
    V3_ONLY_ARGUMENTS,
    convert_output_file,
    detect_version,
    from_v3_kwargs,
    to_v3_output,
    to_v4_output,
)
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
    assert kw["resonance_flag"] is False  # v4 has resonance_flag since #151 was ported


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


# Output files. The v4 output comes from the analytic DBS_synthetic preset; v3
# files are not available here, so the v3 layout is checked through names
# and round trips (comparisons with real v3 output are done outside the suite)
@pytest.fixture(scope="module")
def v4_output(tmp_path_factory):
    kwargs = get_parameters_for_Scotty("DBS_synthetic")
    kwargs.update(len_tau=52, interp_smoothing=0, detailed_analysis_flag=True)
    path = tmp_path_factory.mktemp("v4")
    beam_me_up(**kwargs, output_path=path)
    return next(path.glob("*.h5"))


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind in "USO" or b.dtype.kind in "USO":
        return np.array_equal(a.astype(str), b.astype(str))
    return a.shape == b.shape and np.allclose(a, b, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_output_v4_to_v3_layout(v4_output):
    v4 = xr.open_datatree(v4_output, engine="h5netcdf").load()
    v3 = to_v3_output(v4)
    assert detect_version(v4) == "v4" and detect_version(v3) == "v3"
    assert set(v3.children) == {"inputs", "solver_output", "analysis"}
    for name in ("q_R", "q_Z", "q_zeta", "K_R", "K_Z", "Psi_3D"):
        assert name in v3["solver_output"].ds
    for name in ("poloidal_flux", "loc_b", "loc_m", "theta_m", "delta_theta_m", "b_hat", "K_zeta_initial"):
        assert name in v3["analysis"].ds
    assert bool(v3["inputs"].ds["vacuumLaunch_flag"].values)
    assert_allclose(v3["solver_output"].ds["q_R"], v4["solver_output"].ds["q_R_output"])
    assert_allclose(v3["analysis"].ds["loc_b"], v4["dbs_analysis"].ds["loc_b"])
    # v3's initial position is the first point of the ray
    so = v4["solver_output"].ds
    assert_allclose(v3["inputs"].ds["initial_position"],
                    [so["q_R_output"][0], so["q_zeta_output"][0], so["q_Z_output"][0]])
    assert "dbs_analysis/beam_waist" in v3.attrs["scotty_compat_not_converted"]


def test_output_cylindrical_vectors(v4_output):
    v4 = xr.open_datatree(v4_output, engine="h5netcdf").load()
    v3 = to_v3_output(v4)
    zeta = v4["solver_output"].ds["q_zeta_output"].values
    b_cyl = v3["analysis"].ds["b_hat"].values
    b_cart = v4["analysis"].ds["b_hat_cartesian"].values
    # unit vector, and b_X = b_R cos(zeta) - b_zeta sin(zeta)
    assert_allclose(np.linalg.norm(b_cyl, axis=1), 1.0)
    assert_allclose(b_cyl[:, 0] * np.cos(zeta) - b_cyl[:, 1] * np.sin(zeta), b_cart[:, 0], atol=1e-12)


def test_output_round_trips(v4_output):
    v4 = xr.open_datatree(v4_output, engine="h5netcdf").load()
    v3 = to_v3_output(v4)
    back4 = to_v4_output(v3)
    for group in back4.children:
        for name in back4[group].ds.data_vars:
            if name in v4[group].ds:
                assert _same(back4[group].ds[name].values, v4[group].ds[name].values), f"{group}/{name}"
    back3 = to_v3_output(back4)
    for group in back3.children:
        for name in back3[group].ds.data_vars:
            assert _same(back3[group].ds[name].values, v3[group].ds[name].values), f"{group}/{name}"


def test_output_file_conversion(v4_output, tmp_path):
    assert convert_output_file(v4_output, tmp_path / "as_v3.h5") == "v3"
    assert convert_output_file(tmp_path / "as_v3.h5", tmp_path / "as_v4.h5") == "v4"
    with xr.open_datatree(tmp_path / "as_v4.h5", engine="h5netcdf") as dt:
        assert detect_version(dt) == "v4"
        assert dt.attrs["scotty_compat_converted_from"] == "v3"


def test_quick_cutoff_reproduces_v3_quick_run(tmp_path):
    """v3's quick_run returned (R, Z, |K|, poloidal flux, theta_m) at the
    minimum of |K|; the v3 values here are from v3 (b766f33) on the same case"""
    from scotty.compat import quick_cutoff

    kwargs = get_parameters_for_Scotty("DBS_synthetic")
    kwargs.update(find_B_method="unit-tests", interp_smoothing=0, figure_flag=False, quick_run=True)
    for name in ("geometry", "launch_flag", "boundary_flag"):
        kwargs.pop(name)
    kwargs.update(Psi_BC_flag="continuous", vacuumLaunch_flag=True)
    dt = beam_me_up(**from_v3_kwargs(kwargs), output_path=tmp_path)

    q_R, q_Z, K_norm, polflux, theta_m = quick_cutoff(dt)
    v3 = (1.6940035472950905, -0.18096914886063376, 561.5146915492645, 0.5306117571348025, 0.13255153229421354)
    assert_allclose([q_R, q_Z, K_norm, polflux, theta_m], v3, rtol=2e-4)
    assert quick_cutoff(dt).theta_m == theta_m   # attribute access, as for v3's K_cutoff_data
