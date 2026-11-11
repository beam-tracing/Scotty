import inspect
import logging
import pathlib

import pytest

from scotty.beam_me_up import beam_me_up
from scotty.checks import Parameters
from scotty.init_bruv import get_parameters_for_Scotty


def _check_flux(enter, zero_density, zero_temperature):
    # The checks only use the class's helper methods, so no full Parameters set-up is needed
    return Parameters._check_poloidal_flux_arguments(
        Parameters.__new__(Parameters), enter, zero_density, zero_temperature, False
    )


def test_poloidal_flux_equal_values_accepted():
    assert _check_flux(1.3, 1.3, 1.3) == (1.3, 1.3, 1.3)


def test_poloidal_flux_zero_density_below_enter_names_density():
    with pytest.raises(ValueError) as err:
        _check_flux(1.3, 1.0, 1.3)
    message = str(err.value)
    assert "`poloidal_flux_zero_density` must be greater than or equal to" in message
    assert "= 1.0" in message


def test_poloidal_flux_zero_temperature_below_enter_names_temperature():
    with pytest.raises(ValueError) as err:
        _check_flux(1.3, 1.3, 1.0)
    message = str(err.value)
    assert (
        "`poloidal_flux_zero_temperature` must be greater than or equal to" in message
    )
    assert "`poloidal_flux_zero_density`" not in message
    assert "= 1.0" in message


def _circular_flag(**extras):
    params = Parameters.__new__(Parameters)
    params._unpack_kwargs_for_circular_flux_surfaces(extras=extras)
    return params.circular_flux_surfaces_flag


def test_circular_flux_surfaces_accept_zero_poloidal_field():
    # B_p_a = 0 (no poloidal field) is a valid value, not a missing one
    assert _circular_flag(B_T_axis=1.0, B_p_a=0.0, R_axis=1.5, minor_radius_a=0.5)


def test_circular_flux_surfaces_need_all_or_none():
    assert not _circular_flag()
    with pytest.raises(ValueError):
        _circular_flag(B_T_axis=1.0, R_axis=1.5)


##################################################
# Decisions on PR #164 (C1-C14), 2026-11-02
##################################################


def _synthetic(tmp_path, **kwargs):
    """A quick ray-tracing run of the DBS_synthetic preset"""
    parameters = get_parameters_for_Scotty("DBS_synthetic")
    parameters.update(
        len_tau=12,
        ray_tracing_flag=True,
        console_log_level="WARNING",
        output_path=tmp_path,
    )
    parameters.update(kwargs)
    return parameters


def test_poloidal_flux_zero_temperature_defaults_to_zero_density():
    """C3: None (the default) is poloidal_flux_zero_density"""
    assert _check_flux(1.0, 1.2, None) == (1.0, 1.2, 1.2)
    default = inspect.signature(beam_me_up).parameters["poloidal_flux_zero_temperature"]
    assert default.default is None


def test_poloidal_flux_zero_temperature_given_still_checked():
    """C3: a value given below poloidal_flux_enter still raises"""
    with pytest.raises(ValueError, match="poloidal_flux_zero_temperature"):
        _check_flux(1.3, 1.3, 1.1)


def test_poloidal_flux_zero_temperature_none_in_a_run(tmp_path):
    dt = beam_me_up(
        **_synthetic(
            tmp_path,
            poloidal_flux_zero_density=1.05,
            poloidal_flux_zero_temperature=None,
        )
    )
    assert float(dt["inputs"]["poloidal_flux_zero_temperature"]) == 1.05


@pytest.mark.parametrize(
    ("name", "v3_default"),
    [
        ("interp_smoothing", 0),
        ("find_B_method", "torbeam"),
        ("detailed_analysis_flag", True),
    ],
)
def test_defaults_are_v3s(name, v3_default):
    """C2: the defaults that had changed silently from v3 are v3's again"""
    assert inspect.signature(beam_me_up).parameters[name].default == v3_default


@pytest.mark.parametrize("name", ["delta_X", "delta_Y", "delta_R", "delta_Z"])
def test_default_spatial_steps(name):
    """C12: the default spatial finite-difference step is 1e-5 (v3: 1e-4); the
    wavevector steps are unchanged"""
    parameters = inspect.signature(beam_me_up).parameters
    assert parameters[name].default == 1e-5
    for K_name in ("delta_K_X", "delta_K_Y", "delta_K_R", "delta_K_zeta", "delta_K_Z"):
        assert parameters[K_name].default == 1e-1


@pytest.mark.parametrize("launch_flag", ["vacuum", "Vacuum_analytic", True, None])
def test_launch_flag_must_be_valid(tmp_path, launch_flag):
    """C1: "vacuum" is gone (no alias); the error lists the valid values"""
    with pytest.raises(ValueError) as err:
        beam_me_up(**_synthetic(tmp_path, launch_flag=launch_flag))
    for valid in ("vacuum_analytic", "vacuum_numerical", "plasma"):
        assert valid in str(err.value)


@pytest.mark.parametrize("Psi_BC_flag", [True, False, "None", 1])
def test_Psi_BC_flag_must_be_valid(tmp_path, Psi_BC_flag):
    """C1: exactly "continuous", "discontinuous" or None; v3's booleans raise"""
    with pytest.raises(ValueError, match="`Psi_BC_flag` must be"):
        beam_me_up(**_synthetic(tmp_path, Psi_BC_flag=Psi_BC_flag))


@pytest.mark.parametrize("Psi_BC_flag", ["continuous", "discontinuous"])
def test_vacuum_numerical_refuses_boundary_conditions(tmp_path, Psi_BC_flag):
    """C1: a numerical vacuum launch has no boundary step"""
    with pytest.raises(ValueError, match="vacuum_numerical"):
        beam_me_up(
            **_synthetic(
                tmp_path, launch_flag="vacuum_numerical", Psi_BC_flag=Psi_BC_flag
            )
        )


def test_boundary_flag_is_now_Psi_BC_flag(tmp_path, caplog):
    """C1: the v4 name `boundary_flag` is gone; it is an unused keyword"""
    with caplog.at_level(logging.WARNING):
        dt = beam_me_up(**_synthetic(tmp_path, boundary_flag="discontinuous"))
    assert str(dt["inputs"]["Psi_BC_flag"].values) == "continuous"
    assert "boundary_flag" in caplog.text
    assert "did you mean" in caplog.text


@pytest.mark.parametrize("as_type", [str, pathlib.Path])
def test_paths_accept_str_and_Path(tmp_path, as_type):
    """C4: all four paths can be given as str or pathlib.Path; the output
    folder is created if it does not exist, and the output written there"""
    data = tmp_path / "data"
    data.mkdir()
    output = tmp_path / "new" / "output"
    beam_me_up(
        **_synthetic(
            tmp_path,
            magnetic_data_path=as_type(data),
            ne_data_path=as_type(data),
            Te_data_path=as_type(data),
            output_path=as_type(output),
            output_filename_suffix="_paths",
        )
    )
    assert (output / "scotty_output_paths.h5").is_file()


def test_output_folder_creation_is_logged(tmp_path, caplog):
    output = tmp_path / "created"
    with caplog.at_level(logging.INFO, logger="scotty.checks"):
        Parameters._check_data_path(
            Parameters.__new__(Parameters), "output", str(output)
        )
    assert output.is_dir()
    assert "Creating the folder" in caplog.text


@pytest.mark.parametrize("name", ["magnetic", "ne", "Te"])
def test_missing_data_path_raises(tmp_path, name):
    with pytest.raises(ValueError, match=f"`{name}_data_path`"):
        Parameters._check_data_path(
            Parameters.__new__(Parameters), name, str(tmp_path / "missing")
        )


def test_misspelt_keyword_warns_with_suggestion(caplog):
    """C5: a warning, not a print, suggesting the closest argument name"""
    params = Parameters.__new__(Parameters)
    with caplog.at_level(logging.WARNING, logger="scotty.checks"):
        params._unpack_unused_kwargs(
            extras={"lauch_freq_GHz": 55.0, "poloidal_flux_entr": 1.0, "xyzzy": 1}
        )
    assert "Unused keyword arguments" in caplog.text
    assert "did you mean `launch_freq_GHz`?" in caplog.text
    assert "did you mean `poloidal_flux_enter`?" in caplog.text
    # No suggestion for a name like no argument
    xyzzy_line = next(line for line in caplog.text.splitlines() if "xyzzy" in line)
    assert "did you mean" not in xyzzy_line


def test_unused_keyword_columns_fit_the_longest_name(caplog):
    """The column width is measured on the unused keywords (it used to be
    measured on the dict holding them, whose only key is "extras")"""
    long_name = "a_very_long_misspelt_keyword_argument"
    params = Parameters.__new__(Parameters)
    with caplog.at_level(logging.WARNING, logger="scotty.checks"):
        params._unpack_unused_kwargs(extras={long_name: 1, "short": 2})
    lines = caplog.text.splitlines()
    long_line = next(line for line in lines if long_name in line)
    short_line = next(line for line in lines if "- short" in line)
    # Values start in the same column, after the longest name
    assert long_line.index(" 1") == short_line.index(" 2")
    assert long_line.index(" 1") > long_line.index(long_name) + len(long_name)


##################################################
# Resonance flags (#169, replacing resonance_flag)
##################################################

RESONANCE_FLAG_NAMES = (
    "first_harmonic_flag",
    "second_harmonic_flag",
    "upper_hybrid_flag",
)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("first_harmonic_flag", "stop"),
        ("first_harmonic_flag", "raise"),
        *[
            (name, value)
            for name in RESONANCE_FLAG_NAMES[1:]
            for value in ("stop", "raise", "pass")
        ],
    ],
)
def test_resonance_flags_accept_their_values(name, value):
    params = Parameters.__new__(Parameters)
    assert params._check_resonance_flag(name, value) == value


@pytest.mark.parametrize("value", ["Stop", "", True, False, 2, 0, None])
@pytest.mark.parametrize("name", RESONANCE_FLAG_NAMES)
def test_resonance_flags_refuse_other_values(name, value):
    """Including the old resonance_flag's booleans and integers. The error names
    the flag, lists the values and shows the one given"""
    params = Parameters.__new__(Parameters)
    with pytest.raises(ValueError, match=f"`{name}` must be one of") as error:
        params._check_resonance_flag(name, value)
    for valid in ("'stop'", "'raise'", "'pass'"):
        assert valid in str(error.value)
    assert repr(value) in str(error.value)


def test_first_harmonic_flag_refuses_pass():
    """The cold-plasma Hamiltonian is singular there"""
    params = Parameters.__new__(Parameters)
    with pytest.raises(ValueError, match="can't be 'pass'") as error:
        params._check_resonance_flag("first_harmonic_flag", "pass")
    assert "singular" in str(error.value)


@pytest.mark.parametrize("name", RESONANCE_FLAG_NAMES)
def test_resonance_flags_default_to_stop(name):
    assert inspect.signature(beam_me_up).parameters[name].default == "stop"


def test_resonance_flag_checked_in_a_run(tmp_path):
    with pytest.raises(ValueError, match="`second_harmonic_flag` must be one of"):
        beam_me_up(**_synthetic(tmp_path, second_harmonic_flag="Stop"))
    with pytest.raises(ValueError, match="`first_harmonic_flag` can't be 'pass'"):
        beam_me_up(**_synthetic(tmp_path, first_harmonic_flag="pass"))


@pytest.mark.parametrize("value", [True, False, 0, 2, "stop"])
def test_resonance_flag_raises_that_it_is_replaced(tmp_path, monkeypatch, value):
    """Rather than being ignored as an unused keyword, which would change where
    the ray stops. Raised before anything is traced"""
    import importlib

    def no_tracing(**kwargs):
        raise AssertionError("traced the ray")

    monkeypatch.setattr(
        importlib.import_module("scotty.beam_me_up"), "ray_tracing", no_tracing
    )
    with pytest.raises(ValueError, match="`resonance_flag`") as error:
        beam_me_up(**_synthetic(tmp_path, resonance_flag=value))
    message = str(error.value)
    assert "has been replaced" in message
    for name in (*RESONANCE_FLAG_NAMES, "from_v3_kwargs"):
        assert name in message


def test_check_Psi_output_removed():
    """C13: it warned on every run"""
    import scotty.checks

    assert not hasattr(scotty.checks, "check_Psi_output")


def test_beam_me_up_docstring_covers_every_argument():
    """Item 19: a numpydoc docstring with every argument, and the launch and
    boundary-condition flags' values"""
    doc = beam_me_up.__doc__
    assert "INSERT DOC STRING HERE" not in inspect.getsource(beam_me_up)
    for section in ("Parameters\n    ----------", "Returns\n    -------"):
        assert section in doc
    for name, parameter in inspect.signature(beam_me_up).parameters.items():
        documented = (
            "**kwargs" if parameter.kind == inspect.Parameter.VAR_KEYWORD else name
        )
        assert documented in doc, name
    for value in (
        '"vacuum_analytic"',
        '"vacuum_numerical"',
        '"plasma"',
        '"continuous"',
        '"discontinuous"',
    ):
        assert value in doc


@pytest.mark.parametrize(
    "module",
    [
        "analysis",
        "beam_solver",
        "boundary_conditions",
        "checks",
        "compat",
        "data_io",
        "derivatives",
        "logger",
        "plotting",
    ],
)
def test_licence_header(module):
    """Item 19: the licence header of the other modules"""
    path = pathlib.Path(__file__).parents[1] / "scotty" / f"{module}.py"
    head = path.read_text(encoding="utf-8").splitlines()[:3]
    assert head == [
        "# -*- coding: utf-8 -*-",
        "# Copyright 2018 - 2023, Valerian Hall-Chen and the Scotty contributors",
        "# SPDX-License-Identifier: GPL-3.0",
    ]
