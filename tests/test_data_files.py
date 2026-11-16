"""The data files are given by their paths, with a spline order and smoothing
for each of the magnetic field, the density and the temperature (4.0.0)

Before, ``magnetic_data_path``, ``ne_data_path`` and ``Te_data_path`` were
folders, the file names ended in ``input_filename_suffix``, and one
``interp_order`` and one ``interp_smoothing`` set all three splines. A file
given in ``density_fit_parameters`` always raised "expected filename as second
parameter" (B15, as in v3). ``find_B_method`` "test", "test_notime",
"UDA_saved" and "EFITpp" are removed for now, with ``shot`` and ``equil_time``.
"""

import importlib
import logging
import pathlib

import numpy as np
import pytest

from scotty.beam_me_up import beam_me_up
from scotty.compat import from_v3_kwargs
from scotty.geometry import CircularFluxSurfaceField, EFITField, REMOVED_FIND_B_METHODS
from scotty.init_bruv import get_parameters_for_Scotty
from scotty.profile_fit import SmoothingSplineFit, TanhFit
from scotty.torbeam import Torbeam

PSI = np.linspace(0.0, 1.0, 101)
GRID = np.linspace(0.0, 1.0, 1001)


def write_profile(path, values):
    """A file in the format of `SmoothingSplineFit.from_dat_file`"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"{len(PSI)}\n")
        np.savetxt(f, np.column_stack((np.sqrt(PSI), values)), fmt="%.8e")
    return path


@pytest.fixture
def ne_dat(tmp_path):
    return write_profile(tmp_path / "data" / "ne.dat", 4.0 * (1.0 - PSI**2))


@pytest.fixture
def te_dat(tmp_path):
    return write_profile(tmp_path / "data" / "Te.dat", 1.0 * (1.0 - PSI))


def preset(**changes):
    kwargs = get_parameters_for_Scotty("DBS_synthetic")
    kwargs.pop("density_fit_parameters", None)
    kwargs.update(len_tau=20)
    kwargs.update(changes)
    return kwargs


def run(tmp_path, **changes):
    return beam_me_up(**preset(output_path=tmp_path / "out", **changes))


def same_beam(a, b):
    return np.array_equal(
        a["beam"].ds["width_1"].values, b["beam"].ds["width_1"].values
    ) and np.array_equal(a["solver"].ds["tau"].values, b["solver"].ds["tau"].values)


def write_topfile(path):
    """A TORBEAM topfile of the DBS_synthetic preset's circular flux surfaces"""
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = get_parameters_for_Scotty("DBS_synthetic")
    field = CircularFluxSurfaceField(
        R_axis=kwargs["R_axis"],
        minor_radius_a=kwargs["minor_radius_a"],
        B_T_axis=kwargs["B_T_axis"],
        B_p_a=kwargs["B_p_a"],
        R_points=129,
        Z_points=129,
        grid_buffer_factor=1.1,
    )
    R, Z = np.meshgrid(field.R_coord, field.Z_coord, indexing="ij")
    Torbeam(
        field.R_coord,
        field.Z_coord,
        field.B_R(R, 0, Z),
        field.B_T(R, 0, Z),
        field.B_Z(R, 0, Z),
        field.polflux(R, 0, Z),
    ).write(path)
    return path


##################################################
# Data files by their paths
##################################################


@pytest.mark.parametrize("as_path", [str, pathlib.Path], ids=["str", "Path"])
def test_density_file_by_its_path(tmp_path, ne_dat, as_path):
    named = run(
        tmp_path,
        density_fit_method="smoothing-spline-file",
        ne_data_path=as_path(ne_dat),
    )
    obj = run(
        tmp_path, density_fit_method=SmoothingSplineFit.from_dat_file(1.0, ne_dat)
    )
    assert same_beam(named, obj)
    assert str(named["inputs"]["ne_data_path"].values) == str(ne_dat)


def test_relative_path_is_read_from_the_current_folder(tmp_path, monkeypatch):
    here = write_profile(tmp_path / "here" / "ne.dat", 4.0 * (1.0 - PSI**2))
    monkeypatch.chdir(here.parent)
    relative = run(
        tmp_path, density_fit_method="smoothing-spline-file", ne_data_path="ne.dat"
    )
    obj = run(tmp_path, density_fit_method=SmoothingSplineFit.from_dat_file(1.0, here))
    assert same_beam(relative, obj)


def test_temperature_file_by_its_path(tmp_path, ne_dat, te_dat):
    common = dict(
        relativistic_flag=True,
        density_fit_method="smoothing-spline-file",
        ne_data_path=ne_dat,
    )
    named = run(
        tmp_path,
        temperature_fit_method="smoothing-spline-file",
        Te_data_path=te_dat,
        **common,
    )
    obj = run(
        tmp_path,
        temperature_fit_method=SmoothingSplineFit.from_dat_file(1.0, te_dat),
        **common,
    )
    assert same_beam(named, obj)


def test_topfile_by_its_path(tmp_path):
    topfile = write_topfile(tmp_path / "data" / "topfile_12345")
    dt, field = run(
        tmp_path,
        find_B_method="torbeam",
        magnetic_data_path=topfile,
        return_dt_field=True,
    )
    assert field.interp_order == 5
    assert str(dt["inputs"]["magnetic_data_path"].values) == str(topfile)


@pytest.mark.parametrize(
    ("name", "changes", "file_name"),
    [
        ("magnetic_data_path", {"find_B_method": "torbeam"}, "topfile"),
        ("ne_data_path", {"density_fit_method": "smoothing-spline-file"}, "ne.dat"),
        (
            "Te_data_path",
            {
                "relativistic_flag": True,
                "temperature_fit_method": "smoothing-spline-file",
            },
            "Te.dat",
        ),
    ],
)
def test_folder_raises_with_the_file_name(tmp_path, name, changes, file_name):
    """A v4 script from before 4.0.0 gave the folder: the error says what to give"""
    with pytest.raises(ValueError, match=f"`{name}` = .* is a folder") as error:
        run(tmp_path, **{name: tmp_path, **changes})
    assert str(tmp_path / file_name) in str(error.value)


def test_file_that_is_not_there_raises(tmp_path):
    with pytest.raises(ValueError, match="`ne_data_path` = .*no such file"):
        run(
            tmp_path,
            density_fit_method="smoothing-spline-file",
            ne_data_path=tmp_path / "ne.dat",
        )


def test_file_fit_needs_its_path(tmp_path):
    with pytest.raises(ValueError, match="`ne_data_path` must be given"):
        run(tmp_path, density_fit_method="smoothing-spline-file")


def test_unused_path_gives_a_warning(tmp_path, ne_dat, caplog):
    with caplog.at_level(logging.WARNING, logger="scotty.checks"):
        run(tmp_path, ne_data_path=ne_dat)
    assert "`ne_data_path`" in caplog.text
    assert "is not used" in caplog.text


@pytest.mark.parametrize("quantity", ["density", "temperature"])
def test_file_fit_takes_no_parameters(tmp_path, ne_dat, te_dat, quantity):
    """The path, order and smoothing have their own arguments"""
    changes = dict(
        relativistic_flag=True,
        density_fit_method="smoothing-spline-file",
        ne_data_path=ne_dat,
        temperature_fit_method="smoothing-spline-file",
        Te_data_path=te_dat,
    )
    changes[f"{quantity}_fit_parameters"] = [ne_dat, 5, 0]
    with pytest.raises(ValueError, match=f"`{quantity}_fit_parameters` must not"):
        run(tmp_path, **changes)


##################################################
# One spline order and smoothing for each of B, ne and Te
##################################################


@pytest.mark.parametrize(("order", "smoothing"), [(3, 0.0), (5, 0.5)])
def test_density_spline_settings(tmp_path, ne_dat, order, smoothing):
    named = run(
        tmp_path,
        density_fit_method="smoothing-spline-file",
        ne_data_path=ne_dat,
        interp_order_ne=order,
        interp_smoothing_ne=smoothing,
    )
    obj = run(
        tmp_path,
        density_fit_method=SmoothingSplineFit.from_dat_file(
            1.0, ne_dat, order, smoothing
        ),
    )
    assert same_beam(named, obj)
    assert int(named["inputs"]["interp_order_ne"]) == order
    assert float(named["inputs"]["interp_smoothing_ne"]) == smoothing


def test_spline_settings_are_separate(tmp_path, ne_dat):
    """The order of the density spline does not change the field's, and the
    other way round"""
    topfile = write_topfile(tmp_path / "data" / "topfile")
    captured = []
    beam_me_up_module = importlib.import_module("scotty.beam_me_up")
    make_fit = beam_me_up_module.make_fit

    def spy(*args, **kwargs):
        captured.append(make_fit(*args, **kwargs))
        return captured[-1]

    common = dict(
        find_B_method="torbeam",
        magnetic_data_path=topfile,
        density_fit_method="smoothing-spline-file",
        ne_data_path=ne_dat,
        return_dt_field=True,
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(beam_me_up_module, "make_fit", spy)
        _, field = run(tmp_path, interp_order_ne=3, **common)
        assert (field.interp_order, captured[-1].spline._eval_args[2]) == (5, 3)
        _, field = run(tmp_path, interp_order_magnetic=3, **common)
        assert (field.interp_order, captured[-1].spline._eval_args[2]) == (3, 5)


@pytest.mark.parametrize("name", ["interp_order_magnetic", "interp_order_ne"])
@pytest.mark.parametrize("order", [4, 0, "quartic", True, None])
def test_invalid_order_raises(tmp_path, name, order):
    """It used to be replaced by 5, with a warning only"""
    with pytest.raises(ValueError, match=f"`{name}` must be one of"):
        run(tmp_path, **{name: order})


@pytest.mark.parametrize("smoothing", [-1, np.nan, "0", None])
def test_invalid_smoothing_raises(tmp_path, smoothing):
    with pytest.raises(ValueError, match="`interp_smoothing_ne` must be a number"):
        run(tmp_path, interp_smoothing_ne=smoothing)


def test_order_by_name(tmp_path, ne_dat):
    by_name = run(
        tmp_path,
        density_fit_method="smoothing-spline-file",
        ne_data_path=ne_dat,
        interp_order_ne="cubic",
    )
    assert int(by_name["inputs"]["interp_order_ne"]) == 3


##################################################
# Removed in 4.0.0
##################################################


@pytest.mark.parametrize(
    ("name", "value", "replacement"),
    [
        ("interp_order", 3, "interp_order_magnetic"),
        ("interp_smoothing", 0, "interp_smoothing_magnetic"),
        ("input_filename_suffix", "_1", "ne_data_path"),
        ("shot", 29908, "find_B_method"),
        ("equil_time", 0.2, "find_B_method"),
    ],
)
def test_removed_arguments_raise(tmp_path, name, value, replacement):
    """An unknown keyword only gives a warning; these would change the run"""
    with pytest.raises(ValueError, match=f"`{name}` .*was removed") as error:
        run(tmp_path, **{name: value})
    assert replacement in str(error.value)
    assert "from_v3_kwargs" in str(error.value)


@pytest.mark.parametrize("name", ["test", "test_notime", "UDA_saved", "EFITpp"])
def test_removed_find_B_methods_raise(tmp_path, name):
    assert name.lower() in REMOVED_FIND_B_METHODS
    with pytest.raises(ValueError, match="is deprecated, and removed for now"):
        run(tmp_path, find_B_method=name)
    with pytest.raises(ValueError, match="is deprecated, and removed for now"):
        get_parameters_for_Scotty("DBS_NSTX_MAST", 55.0, find_B_method=name)


def test_EFITField_is_deprecated():
    R = np.linspace(1.0, 2.0, 11)
    Z = np.linspace(-0.5, 0.5, 11)
    psi = np.add.outer((R - 1.5) ** 2, Z**2) / 0.25
    with pytest.warns(DeprecationWarning, match="EFITField"):
        EFITField(R, Z, np.ones(11), psi, 0.0, 1.0)


def test_make_fit_moved_to_profile_fit():
    from scotty.profile_fit import make_fit

    assert importlib.import_module("scotty.beam_me_up").make_fit is make_fit


##################################################
# Presets
##################################################


def test_torbeam_preset_gives_the_paths_of_the_files(tmp_path):
    kwargs = get_parameters_for_Scotty(
        "DBS_UCLA_MAST-U", 55.0, find_B_method="torbeam", data_path=tmp_path
    )
    assert kwargs["magnetic_data_path"] == tmp_path / "topfile"
    assert kwargs["ne_data_path"] == tmp_path / "ne.dat"
    assert kwargs["density_fit_method"] == "smoothing-spline-file"


def test_preset_saved_density_fit_without_a_field_file():
    """`shot` and `equil_time` choose a saved density fit; they are not
    passed to `beam_me_up`"""
    kwargs = get_parameters_for_Scotty(
        "DBS_NSTX_MAST", 55.0, find_ne_method="tanh", shot=29908, equil_time=0.16
    )
    assert isinstance(kwargs["density_fit_method"], TanhFit)
    assert "shot" not in kwargs
    assert "equil_time" not in kwargs


##################################################
# v3 scripts
##################################################


def test_v3_folders_become_files(tmp_path):
    kw = from_v3_kwargs(
        {
            "magnetic_data_path": tmp_path,
            "ne_data_path": tmp_path,
            "Te_data_path": tmp_path,
            "input_filename_suffix": "_45000",
            "interp_order": 3,
            "interp_smoothing": 0.5,
            "relativistic_flag": True,
        }
    )
    assert kw["magnetic_data_path"] == tmp_path / "topfile_45000"
    assert kw["ne_data_path"] == tmp_path / "ne_45000.dat"
    assert kw["Te_data_path"] == tmp_path / "Te_45000.dat"
    for quantity in ("magnetic", "ne", "Te"):
        assert kw[f"interp_order_{quantity}"] == 3
        assert kw[f"interp_smoothing_{quantity}"] == 0.5
    for name in ("input_filename_suffix", "interp_order", "interp_smoothing"):
        assert name not in kw
    assert from_v3_kwargs(kw) == kw


def test_v3_omfit_file():
    kw = from_v3_kwargs({"find_B_method": "omfit", "magnetic_data_path": "."})
    assert kw["magnetic_data_path"] == pathlib.Path("topfile.json")


def test_v3_shot_and_time_are_dropped():
    kw = from_v3_kwargs({"shot": 29908, "equil_time": 0.2})
    assert "shot" not in kw
    assert "equil_time" not in kw


@pytest.mark.parametrize("as_path", [str, pathlib.Path], ids=["str", "Path"])
def test_v3_script_with_a_file_in_its_parameters(ne_dat, as_path):
    """v3 refused this form (B15); v3 guessed "tanh" from three parameters"""
    kw = from_v3_kwargs({"density_fit_parameters": [as_path(ne_dat), 3, 0.5]})
    assert kw["density_fit_method"] == "smoothing-spline-file"
    assert kw["ne_data_path"] == ne_dat
    assert (kw["interp_order_ne"], kw["interp_smoothing_ne"]) == (3, 0.5)
    assert "density_fit_parameters" not in kw


def test_v3_script_reads_the_same_files(tmp_path):
    """A v3 script with folders and a suffix gives the v4 run with the paths"""
    write_profile(tmp_path / "data" / "ne_45000.dat", 4.0 * (1.0 - PSI**2))
    write_topfile(tmp_path / "data" / "topfile_45000")
    v4 = preset(
        output_path=tmp_path / "out",
        find_B_method="torbeam",
        magnetic_data_path=tmp_path / "data" / "topfile_45000",
        density_fit_method="smoothing-spline-file",
        ne_data_path=tmp_path / "data" / "ne_45000.dat",
        interp_order_magnetic=3,
        interp_order_ne=3,
    )
    v3 = {
        name: value
        for name, value in v4.items()
        if not name.startswith("interp_order") and not name.endswith("data_path")
    }
    v3.update(
        magnetic_data_path=tmp_path / "data",
        ne_data_path=tmp_path / "data",
        input_filename_suffix="_45000",
        interp_order=3,
    )
    kw = from_v3_kwargs(v3)
    # The other v3 defaults that from_v3_kwargs sets, for both runs
    extra = {name: value for name, value in kw.items() if name not in v4}
    assert not [name for name in extra if "data_path" in name or "interp" in name]
    assert same_beam(beam_me_up(**kw), beam_me_up(**v4, **extra))
