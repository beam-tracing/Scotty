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
    for name in ("geometry", "launch_flag", "Psi_BC_flag"):
        kwargs.pop(name)
    kwargs.update(
        vacuumLaunch_flag=True, vacuum_propagation_flag=True, Psi_BC_flag="continuous"
    )
    return kwargs


@pytest.mark.parametrize(
    ("v3_flags", "launch_flag"),
    [
        (
            {"vacuumLaunch_flag": True, "vacuum_propagation_flag": True},
            "vacuum_analytic",
        ),
        (
            {"vacuumLaunch_flag": True, "vacuum_propagation_flag": False},
            "vacuum_numerical",
        ),
        # v3's beam_me_up defaulted to vacuum_propagation_flag = False
        ({"vacuumLaunch_flag": True}, "vacuum_numerical"),
        ({"vacuum_propagation_flag": True}, "vacuum_analytic"),
        ({}, "vacuum_numerical"),
        ({"vacuumLaunch_flag": False}, "plasma"),
        ({"vacuumLaunch_flag": False, "vacuum_propagation_flag": True}, "plasma"),
    ],
)
def test_launch_flags_converted(v3_flags, launch_flag, caplog):
    kw = from_v3_kwargs({**v3_flags, "Psi_BC_flag": None})
    assert kw["launch_flag"] == launch_flag
    assert kw["geometry"] == "cylindrical"
    assert not set(V3_ONLY_ARGUMENTS) & set(kw)
    # The mapping is stated in a warning (once per message)
    if caplog.records:
        assert f"`launch_flag` = '{launch_flag}'" in caplog.text


@pytest.mark.parametrize("Psi_BC_flag", ["continuous", "discontinuous", None, True])
def test_Psi_BC_flag_passed_through(Psi_BC_flag):
    """Same name and values in v4. v3's deprecated booleans are passed on
    unchanged, and v4 refuses them"""
    kw = from_v3_kwargs({"vacuum_propagation_flag": True, "Psi_BC_flag": Psi_BC_flag})
    assert kw["Psi_BC_flag"] is Psi_BC_flag


def test_unset_arguments_get_v3_defaults():
    kw = from_v3_kwargs({"poloidal_flux_enter": 1.2})
    assert kw["launch_flag"] == "vacuum_numerical"
    assert kw["Psi_BC_flag"] is None
    # v3's spatial steps (upstream/master feb9499); v4's default is 1e-5
    assert kw["delta_R"] == -1e-4
    assert kw["delta_Z"] == 1e-4
    # v3's defaults are v4's again, so they are not set
    for name in ("interp_smoothing", "find_B_method", "detailed_analysis_flag"):
        assert name not in kw
    # v3's 1.01, raised to poloidal_flux_enter as v4 requires
    assert kw["poloidal_flux_zero_density"] == 1.2
    assert kw["poloidal_flux_zero_temperature"] == 1.2


def test_explicit_arguments_kept_and_idempotent():
    given = {
        "geometry": "cylindrical",
        "launch_flag": "plasma",
        "Psi_BC_flag": "discontinuous",
        "interp_smoothing": 3,
        "poloidal_flux_zero_density": 1.1,
        "vacuum_propagation_flag": True,
    }
    kw = from_v3_kwargs(given)
    for name in (
        "launch_flag",
        "Psi_BC_flag",
        "interp_smoothing",
        "poloidal_flux_zero_density",
    ):
        assert kw[name] == given[name]
    assert from_v3_kwargs(kw) == kw


def test_v3_tolerances(caplog):
    # v3's defaults (upstream/master feb9499); v4's are 1e-4, 1e-7
    kw = from_v3_kwargs({})
    assert kw["rtol"] == 1e-3
    assert kw["atol"] == 1e-6
    kw = from_v3_kwargs({"rtol": 1e-6, "atol": 1e-9})
    assert kw["rtol"] == 1e-6
    assert kw["atol"] == 1e-9


def test_v3_len_tau():
    # v3's default (upstream/master feb9499); v4's is 4002
    assert from_v3_kwargs({})["len_tau"] == 102
    assert from_v3_kwargs({"len_tau": 52})["len_tau"] == 52


def test_explicit_spatial_steps_kept(caplog):
    kw = from_v3_kwargs({"delta_R": 2e-5, "delta_Z": -3e-5})
    assert kw["delta_R"] == 2e-5
    assert kw["delta_Z"] == -3e-5
    assert "`delta_R`" not in caplog.text
    assert "`delta_Z`" not in caplog.text


def test_unset_spatial_steps_warned(caplog, monkeypatch):
    import scotty.compat

    monkeypatch.setattr(scotty.compat, "_warned", set())
    from_v3_kwargs({})
    assert "set `delta_R` = -0.0001 (v3's default" in caplog.text
    assert "set `delta_Z` = 0.0001 (v3's default" in caplog.text


@pytest.mark.parametrize("Psi_BC_flag", ["continuous", "discontinuous"])
@pytest.mark.parametrize(
    "v3_flags",
    [
        {"vacuumLaunch_flag": True, "vacuum_propagation_flag": False},
        {},  # v3's defaults
    ],
)
def test_Psi_BC_flag_dropped_for_vacuum_numerical(
    v3_flags, Psi_BC_flag, caplog, monkeypatch
):
    """v3 ignored Psi_BC_flag when it propagated numerically through the
    vacuum; v4 refuses it with "vacuum_numerical", so it is set to None"""
    import scotty.compat

    monkeypatch.setattr(scotty.compat, "_warned", set())
    kw = from_v3_kwargs({**v3_flags, "Psi_BC_flag": Psi_BC_flag})
    assert kw["launch_flag"] == "vacuum_numerical"
    assert kw["Psi_BC_flag"] is None
    assert f"dropped `Psi_BC_flag` = '{Psi_BC_flag}'" in caplog.text
    assert from_v3_kwargs(kw) == kw


def test_Psi_BC_flag_dropped_is_accepted_by_v4():
    """The converted arguments pass v4's check, which refuses a
    Psi_BC_flag with "vacuum_numerical\" """
    from scotty.checks import Parameters

    kw = from_v3_kwargs({"Psi_BC_flag": "continuous"})
    params = Parameters.__new__(Parameters)
    assert params._check_Psi_BC_flag(kw["Psi_BC_flag"], kw["launch_flag"]) is None
    with pytest.raises(ValueError, match="must be None"):
        params._check_Psi_BC_flag("continuous", "vacuum_numerical")


def test_output_path_and_quick_run():
    kw = from_v3_kwargs({"output_path": "runs/case1/", "quick_run": True})
    assert kw["output_path"] == pathlib.Path("runs/case1")
    assert kw["ray_tracing_flag"] is True


RESONANCE_FLAG_NAMES = (
    "first_harmonic_flag",
    "second_harmonic_flag",
    "upper_hybrid_flag",
)


@pytest.mark.parametrize("value", [True, np.bool_(True)])
def test_resonance_flag_True_is_stop_everywhere(value):
    kw = from_v3_kwargs({"resonance_flag": value})
    assert "resonance_flag" not in kw
    assert [kw[name] for name in RESONANCE_FLAG_NAMES] == ["stop", "stop", "stop"]


@pytest.mark.parametrize("value", [False, np.bool_(False)])
def test_resonance_flag_False_raises_at_the_first_harmonic(value, monkeypatch, caplog):
    """v3's False carried on past the first harmonic and the upper hybrid
    resonance; v4 can't, so it raises there, and the warning says so"""
    import scotty.compat

    # Each message is warned about once per process
    monkeypatch.setattr(scotty.compat, "_warned", set())
    with caplog.at_level("WARNING", logger="scotty.compat"):
        kw = from_v3_kwargs({"resonance_flag": value})
    assert "resonance_flag" not in kw
    assert [kw[name] for name in RESONANCE_FLAG_NAMES] == ["raise", "pass", "raise"]
    assert "`resonance_flag` = False" in caplog.text
    assert "`first_harmonic_flag` = 'raise'" in caplog.text
    assert "`upper_hybrid_flag` = 'raise'" in caplog.text
    assert "where v3 carried on past them" in caplog.text


def test_resonance_flag_keeps_flags_given():
    kw = from_v3_kwargs({"resonance_flag": False, "second_harmonic_flag": "stop"})
    assert [kw[name] for name in RESONANCE_FLAG_NAMES] == ["raise", "stop", "raise"]
    kw = from_v3_kwargs({"resonance_flag": True, "upper_hybrid_flag": "raise"})
    assert [kw[name] for name in RESONANCE_FLAG_NAMES] == ["stop", "stop", "raise"]


@pytest.mark.parametrize("value", [2, 0, "stop", None])
def test_resonance_flag_not_a_bool_is_left_for_v4_to_refuse(value):
    """v3's flag was a bool; anything else is passed on, and v4 raises on it"""
    kw = from_v3_kwargs({"resonance_flag": value})
    assert kw["resonance_flag"] == value
    assert not set(RESONANCE_FLAG_NAMES) & set(kw)


def test_plasma_launch_K_cylindrical_to_cartesian():
    launch_position = np.array([1.8, 0.3, 0.1])
    K_cyl = np.array([-900.0, 50.0, 20.0])
    kw = from_v3_kwargs(
        {
            "vacuumLaunch_flag": False,
            "launch_position": launch_position,
            "plasmaLaunch_K": K_cyl,
            "plasmaLaunch_Psi_3D_lab_Cartesian": np.eye(3),
        }
    )
    # The inverse takes the position in Cartesian
    q_cart = find_q_labframe_cyl_to_cart(launch_position)
    K_back = find_K_labframe_cart_to_cyl(kw["plasmaLaunch_K_cartesian"], q_cart)
    assert_allclose(K_back, K_cyl)
    assert_allclose(kw["plasmaLaunch_Psi_3D_lab_cartesian"], np.eye(3))


def test_v3_script_gives_same_result_as_v4(tmp_path):
    """The v3 script leaves the spatial steps and tolerances unset, so
    from_v3_kwargs sets v3's; the v4 run passes them explicitly"""
    v4 = get_parameters_for_Scotty("DBS_synthetic")
    v4.update(
        len_tau=52,
        interp_smoothing=0,
        delta_R=-1e-4,
        delta_Z=1e-4,
        rtol=1e-3,
        atol=1e-6,
    )
    v3_script = v3_style(v4)
    for name in ("delta_R", "delta_Z", "rtol", "atol"):
        del v3_script[name]
    v3 = from_v3_kwargs(v3_script)
    assert (v3["delta_R"], v3["delta_Z"]) == (-1e-4, 1e-4)
    assert (v3["rtol"], v3["atol"]) == (1e-3, 1e-6)

    out_v4 = beam_me_up(**v4, output_path=tmp_path, output_filename_suffix="_v4")
    out_v3 = beam_me_up(**v3, output_path=tmp_path, output_filename_suffix="_v3")

    for group, name in (
        ("solver", "q_R_output"),
        ("solver", "K_R_output"),
        ("beam", "theta_m"),
    ):
        assert_allclose(out_v3[group][name], out_v4[group][name])


def test_v3_arguments_point_to_the_converter(caplog):
    from scotty.checks import Parameters

    params = Parameters.__new__(Parameters)
    params._unpack_unused_kwargs(
        extras={"vacuum_propagation_flag": True, "vacuumLaunch_flag": True}
    )
    assert "scotty.compat.from_v3_kwargs" in caplog.text
    assert "vacuum_propagation_flag" in caplog.text
    assert "vacuumLaunch_flag" in caplog.text


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
    return a.shape == b.shape and np.allclose(
        a, b, rtol=1e-12, atol=1e-12, equal_nan=True
    )


def test_output_v4_to_v3_layout(v4_output):
    v4 = xr.open_datatree(v4_output, engine="h5netcdf").load()
    v3 = to_v3_output(v4)
    assert detect_version(v4) == "v4" and detect_version(v3) == "v3"
    assert set(v3.children) == {"inputs", "solver_output", "analysis"}
    for name in ("q_R", "q_Z", "q_zeta", "K_R", "K_Z", "Psi_3D"):
        assert name in v3["solver_output"].ds
    for name in (
        "poloidal_flux",
        "loc_b",
        "loc_m",
        "theta_m",
        "delta_theta_m",
        "b_hat",
        "K_zeta_initial",
    ):
        assert name in v3["analysis"].ds
    assert bool(v3["inputs"].ds["vacuumLaunch_flag"].values)
    # DBS_synthetic is a "vacuum_analytic" launch
    assert bool(v3["inputs"].ds["vacuum_propagation_flag"].values)
    assert str(v3["inputs"].ds["Psi_BC_flag"].values) == "continuous"
    assert_allclose(v3["solver_output"].ds["q_R"], v4["solver"].ds["q_R_output"])
    # v3's localisation pieces are v4's filter-function pieces
    for v3_name, v4_name in (
        ("loc_b", "filterfun_b"),
        ("loc_m", "filterfun_m"),
        ("loc_p", "filterfun_p"),
        ("loc_r", "filterfun_r"),
        ("loc_s", "delta_ne_sq_-13_3"),
    ):
        assert_allclose(v3["analysis"].ds[v3_name], v4["diagnostics/dbs"].ds[v4_name])
    # v3's initial position is the first point of the ray
    so = v4["solver"].ds
    assert_allclose(
        v3["inputs"].ds["initial_position"],
        [so["q_R_output"][0], so["q_zeta_output"][0], so["q_Z_output"][0]],
    )
    assert "dbs_analysis/beam_waist" in v3.attrs["scotty_compat_not_converted"]


def test_output_cylindrical_vectors(v4_output):
    v4 = xr.open_datatree(v4_output, engine="h5netcdf").load()
    v3 = to_v3_output(v4)
    zeta = v4["solver"].ds["q_zeta_output"].values
    b_cyl = v3["analysis"].ds["b_hat"].values
    b_cart = v4["beam"].ds["b_hat_cartesian"].values
    # unit vector, and b_X = b_R cos(zeta) - b_zeta sin(zeta)
    assert_allclose(np.linalg.norm(b_cyl, axis=1), 1.0)
    assert_allclose(
        b_cyl[:, 0] * np.cos(zeta) - b_cyl[:, 1] * np.sin(zeta),
        b_cart[:, 0],
        atol=1e-12,
    )


def test_output_round_trips(v4_output):
    v4 = xr.open_datatree(v4_output, engine="h5netcdf").load()
    v3 = to_v3_output(v4)
    back4 = to_v4_output(v3)
    for group in back4.children:
        for name in back4[group].ds.data_vars:
            if name in v4[group].ds:
                assert _same(
                    back4[group].ds[name].values, v4[group].ds[name].values
                ), f"{group}/{name}"
    back3 = to_v3_output(back4)
    for group in back3.children:
        for name in back3[group].ds.data_vars:
            assert _same(
                back3[group].ds[name].values, v3[group].ds[name].values
            ), f"{group}/{name}"


def test_output_file_conversion(v4_output, tmp_path):
    assert convert_output_file(v4_output, tmp_path / "as_v3.h5") == "v3"
    assert convert_output_file(tmp_path / "as_v3.h5", tmp_path / "as_v4.h5") == "v4"
    with xr.open_datatree(tmp_path / "as_v4.h5", engine="h5netcdf") as dt:
        assert detect_version(dt) == "v4"
        assert dt.attrs["scotty_compat_converted_from"] == "v3"


def test_quick_cutoff_reproduces_v3_quick_run(tmp_path):
    """v3's quick_run returned (R, Z, |K|, poloidal flux, theta_m) at the
    minimum of |K|; the v3 values here are from v3 (upstream/master feb9499) on
    the same case, on the analytical field (find_B_method="analytical"), at
    rtol = 1e-6 in both. At rtol = 1e-3 v4's ray cut-off is 1e-3
    from the converged value and v3's 4e-6 (open, with where the saved beam
    ends); at rtol = 1e-6 they agree to 5e-7"""
    from scotty.compat import quick_cutoff

    kwargs = get_parameters_for_Scotty("DBS_synthetic")
    kwargs.update(
        interp_smoothing=0,
        figure_flag=False,
        quick_run=True,
        rtol=1e-6,
    )
    for name in ("geometry", "launch_flag", "Psi_BC_flag"):
        kwargs.pop(name)
    # v3's DBS_synthetic preset; from_v3_kwargs sets v3's default spatial steps
    kwargs.update(
        Psi_BC_flag="continuous",
        vacuumLaunch_flag=True,
        vacuum_propagation_flag=True,
    )
    dt = beam_me_up(**from_v3_kwargs(kwargs), output_path=tmp_path)
    assert (
        abs(float(dt["inputs"]["delta_Z"])) == 1e-4
    )  # v3's step, sign from auto_delta_sign

    q_R, q_Z, K_norm, polflux, theta_m = quick_cutoff(dt)
    v3 = (
        1.7901640559487946,
        -0.16710217160140856,
        444.9064715898386,
        0.4484732604742471,
        0.06686830330882636,
    )
    assert_allclose([q_R, q_Z, K_norm, polflux, theta_m], v3, rtol=2e-4)
    assert (
        quick_cutoff(dt).theta_m == theta_m
    )  # attribute access, as for v3's K_cutoff_data


def _minimal_v3_tree(vacuumLaunch_flag, vacuum_propagation_flag=None):
    """The smallest tree the converter reads as v3: the ray, and the launch
    flags in the inputs"""
    tau = np.linspace(0, 1, 3)
    inputs = {"vacuumLaunch_flag": vacuumLaunch_flag, "Psi_BC_flag": "None"}
    if vacuum_propagation_flag is not None:
        inputs["vacuum_propagation_flag"] = vacuum_propagation_flag
    solver_output = xr.Dataset(
        {name: ("tau", np.zeros(3)) for name in ("q_R", "q_zeta", "q_Z")},
        coords={"tau": tau},
    )
    return xr.DataTree.from_dict(
        {
            "/inputs": xr.Dataset(inputs),
            "/solver_output": solver_output,
            "/analysis": xr.Dataset(coords={"tau": tau}),
        }
    )


@pytest.mark.parametrize(
    ("vacuumLaunch_flag", "vacuum_propagation_flag", "launch_flag"),
    [
        (True, True, "vacuum_analytic"),
        (True, False, "vacuum_numerical"),
        (True, None, "vacuum_numerical"),  # not stored: v3's default, False
        (False, True, "plasma"),
    ],
)
def test_output_launch_flag_from_v3(
    vacuumLaunch_flag, vacuum_propagation_flag, launch_flag
):
    v4 = to_v4_output(_minimal_v3_tree(vacuumLaunch_flag, vacuum_propagation_flag))
    assert str(v4["inputs"].ds["launch_flag"].values) == launch_flag
    assert str(v4["inputs"].ds["Psi_BC_flag"].values) == "None"
    # Consumed into launch_flag, so not reported as unconverted
    assert "vacuum_propagation_flag" not in v4.attrs["scotty_compat_not_converted"]


@pytest.mark.parametrize(
    ("launch_flag", "vacuumLaunch_flag", "vacuum_propagation_flag"),
    [
        ("vacuum_analytic", True, True),
        ("vacuum_numerical", True, False),
        ("plasma", False, False),
        ("vacuum", True, True),  # v4 files before 4.0.0.dev0
    ],
)
def test_output_launch_flag_to_v3(
    v4_output, launch_flag, vacuumLaunch_flag, vacuum_propagation_flag
):
    v4 = xr.open_datatree(v4_output, engine="h5netcdf").load()
    v4["inputs"] = xr.DataTree(
        v4["inputs"].to_dataset().assign(launch_flag=launch_flag)
    )
    v3 = to_v3_output(v4)
    assert bool(v3["inputs"].ds["vacuumLaunch_flag"].values) is vacuumLaunch_flag
    assert (
        bool(v3["inputs"].ds["vacuum_propagation_flag"].values)
        is vacuum_propagation_flag
    )


def test_output_old_boundary_flag_read(v4_output):
    """v4 files before 4.0.0.dev0 stored Psi_BC_flag as inputs/boundary_flag"""
    v4 = xr.open_datatree(v4_output, engine="h5netcdf").load()
    inputs = v4["inputs"].to_dataset()
    inputs = inputs.rename({"Psi_BC_flag": "boundary_flag"})
    v4["inputs"] = xr.DataTree(inputs)
    v3 = to_v3_output(v4)
    assert str(v3["inputs"].ds["Psi_BC_flag"].values) == "continuous"
