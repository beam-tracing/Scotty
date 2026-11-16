"""A profile file given by its path in the parameters of "smoothing-spline-file"

Before this fix, ``density_fit_parameters=[path, order, smoothing]`` always
raised "expected filename as second parameter" (as in v3), so only the default
file ``ne{suffix}.dat`` in ``ne_data_path`` could be read.
"""

import pathlib

import numpy as np
import pytest

from scotty.beam_me_up import beam_me_up
from scotty.compat import from_v3_kwargs
from scotty.init_bruv import get_parameters_for_Scotty
from scotty.profile_fit import SmoothingSplineFit, profile_fit

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


def run(tmp_path, **changes):
    kwargs = get_parameters_for_Scotty("DBS_synthetic")
    kwargs.pop("density_fit_parameters", None)
    kwargs.update(len_tau=20, output_path=tmp_path / "out")
    kwargs.update(changes)
    return beam_me_up(**kwargs)


def same_beam(a, b):
    return np.array_equal(
        a["beam"].ds["width_1"].values, b["beam"].ds["width_1"].values
    ) and np.array_equal(a["solver"].ds["tau"].values, b["solver"].ds["tau"].values)


@pytest.mark.parametrize("as_path", [str, pathlib.Path], ids=["str", "Path"])
def test_named_file_same_as_default_file_and_object(tmp_path, ne_dat, as_path):
    named = run(
        tmp_path,
        density_fit_method="smoothing-spline-file",
        density_fit_parameters=[as_path(ne_dat), 5, 0],
    )
    default = run(
        tmp_path, density_fit_method="smoothing-spline-file", ne_data_path=ne_dat.parent
    )
    obj = run(
        tmp_path, density_fit_method=SmoothingSplineFit.from_dat_file(1.0, ne_dat)
    )
    assert same_beam(named, default)
    assert same_beam(named, obj)


def test_relative_path_is_read_from_the_current_folder(tmp_path, monkeypatch):
    """Not from ``ne_data_path``: that is only for the default file"""
    here = write_profile(tmp_path / "here" / "ne.dat", 4.0 * (1.0 - PSI**2))
    write_profile(tmp_path / "elsewhere" / "ne.dat", 2.0 * (1.0 - PSI**2))
    monkeypatch.chdir(here.parent)

    fit = profile_fit("smoothing-spline-file", 1.0, ["ne.dat"])
    assert np.array_equal(fit(GRID), SmoothingSplineFit.from_dat_file(1.0, here)(GRID))

    relative = run(
        tmp_path,
        density_fit_method="smoothing-spline-file",
        density_fit_parameters=["ne.dat"],
        ne_data_path=tmp_path / "elsewhere",
    )
    obj = run(tmp_path, density_fit_method=SmoothingSplineFit.from_dat_file(1.0, here))
    assert same_beam(relative, obj)


def test_path_alone_uses_order_5_and_no_smoothing(ne_dat):
    fit = profile_fit("smoothing-spline-file", 1.0, [ne_dat])
    expected = SmoothingSplineFit.from_dat_file(1.0, ne_dat, 5, 0.0)
    assert np.array_equal(fit(GRID), expected(GRID))
    # The order is passed on: a cubic spline differs
    cubic = profile_fit("smoothing-spline-file", 1.0, [ne_dat, 3, 0.0])
    assert not np.array_equal(fit(GRID), cubic(GRID))


def test_named_temperature_file_in_a_relativistic_run(tmp_path, ne_dat):
    te_dat = write_profile(tmp_path / "data" / "Te.dat", 1.0 * (1.0 - PSI))
    common = dict(
        relativistic_flag=True, temperature_fit_method="smoothing-spline-file"
    )
    named = run(tmp_path, temperature_fit_parameters=[te_dat, 5, 0], **common)
    default = run(tmp_path, Te_data_path=te_dat.parent, **common)
    assert same_beam(named, default)


@pytest.mark.parametrize(
    "parameters, got",
    [([4.0, 5, 0], "got 4.0"), ([], "got no parameters")],
    ids=["number", "empty"],
)
def test_first_parameter_must_be_a_path(parameters, got):
    with pytest.raises(TypeError, match=f"must be the path of the file.*{got}"):
        profile_fit("smoothing-spline-file", 1.0, parameters)


@pytest.mark.parametrize("as_path", [str, pathlib.Path], ids=["str", "Path"])
def test_v3_script_with_a_file_in_its_parameters(ne_dat, as_path):
    """v3 guessed "tanh" from the three parameters, and failed"""
    kw = from_v3_kwargs({"density_fit_parameters": [as_path(ne_dat), 5, 0]})
    assert kw["density_fit_method"] == "smoothing-spline-file"


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_no_method_with_a_path_guesses_the_file(ne_dat):
    fit = profile_fit(None, 1.0, [ne_dat, 5, 0])
    assert isinstance(fit, SmoothingSplineFit)
