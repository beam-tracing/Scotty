"""The sources of the field and the profiles are required (no defaults), and
the beam and DBS analyses can be turned off"""

import inspect

import numpy as np
import pytest

from scotty.beam_me_up import beam_me_up
from scotty.compat import from_v3_kwargs, to_v3_output
from scotty.data_io import from_output_tree
from scotty.init_bruv import get_parameters_for_Scotty


def synthetic(tmp_path, **changes):
    kwargs = get_parameters_for_Scotty("DBS_synthetic")
    kwargs.update(len_tau=20, output_path=tmp_path)
    kwargs.update(changes)
    return kwargs


# ------------------------------------------------------- required methods


@pytest.mark.parametrize("name", ["find_B_method", "density_fit_method"])
def test_methods_have_no_default(name):
    parameter = inspect.signature(beam_me_up).parameters[name]
    assert parameter.default is inspect.Parameter.empty


@pytest.mark.parametrize("name", ["find_B_method", "density_fit_method"])
def test_missing_method_is_a_TypeError(tmp_path, name):
    kwargs = synthetic(tmp_path)
    del kwargs[name]
    with pytest.raises(TypeError, match=name):
        beam_me_up(**kwargs)


@pytest.mark.parametrize("name", ["find_B_method", "density_fit_method"])
def test_None_method_is_refused(tmp_path, name):
    with pytest.raises(ValueError, match=f"`{name}` must be given"):
        beam_me_up(**synthetic(tmp_path, **{name: None}))


def test_relativistic_needs_a_temperature(tmp_path):
    kwargs = synthetic(tmp_path, relativistic_flag=True)
    kwargs.pop("temperature_fit_method", None)
    with pytest.raises(ValueError, match="needs `temperature_fit_method`"):
        beam_me_up(**kwargs)


def test_temperature_stays_optional_without_relativity():
    parameter = inspect.signature(beam_me_up).parameters["temperature_fit_method"]
    assert parameter.default is None


def test_presets_give_both_methods():
    kwargs = get_parameters_for_Scotty("DBS_synthetic")
    assert kwargs["find_B_method"] is not None
    assert kwargs["density_fit_method"] is not None


@pytest.mark.parametrize(
    "parameters, expected",
    [(None, "smoothing-spline-file"), ([4.0, 1.0], "quadratic")],
)
def test_v3_scripts_get_v3s_methods(parameters, expected):
    v3 = {"density_fit_parameters": parameters} if parameters is not None else {}
    kw = from_v3_kwargs(v3)
    assert kw["find_B_method"] == "torbeam"
    assert kw["density_fit_method"] == expected


def test_v3_relativistic_script_gets_the_temperature_file():
    kw = from_v3_kwargs({"relativistic_flag": True})
    assert kw["temperature_fit_method"] == "smoothing-spline-file"


def test_v3_given_methods_kept():
    kw = from_v3_kwargs({"find_B_method": "omfit", "density_fit_method": "tanh"})
    assert kw["find_B_method"] == "omfit"
    assert kw["density_fit_method"] == "tanh"


# ------------------------------------------------------------ analysis flags


def test_analysis_flags_default_on():
    parameters = inspect.signature(beam_me_up).parameters
    assert parameters["analysis_flag"].default is True
    assert parameters["dbs_analysis_flag"].default is True


def test_dbs_analysis_needs_the_beam_analysis(tmp_path):
    with pytest.raises(ValueError, match="needs `analysis_flag=True`"):
        beam_me_up(**synthetic(tmp_path, analysis_flag=False))


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """The same run with both analyses, with the beam analysis only, and with
    neither"""
    out = {}
    for label, flags in {
        "both": {},
        "beam": {"dbs_analysis_flag": False},
        "none": {"analysis_flag": False, "dbs_analysis_flag": False},
    }.items():
        path = tmp_path_factory.mktemp(label)
        out[label] = beam_me_up(**synthetic(path, **flags))
    return out


def test_groups_follow_the_flags(runs):
    assert {"inputs", "solver", "beam", "diagnostics"} <= set(runs["both"].children)
    assert set(runs["beam"].children) == {"inputs", "solver", "beam"}
    assert set(runs["none"].children) == {"inputs", "solver"}


def test_flags_stored_in_the_inputs(runs):
    inputs = runs["beam"]["inputs"]
    assert bool(inputs["analysis_flag"]) is True
    assert bool(inputs["dbs_analysis_flag"]) is False


def test_solver_output_unchanged_by_the_flags(runs):
    q = runs["both"]["solver"]["q_output_cartesian"].values
    for label in ("beam", "none"):
        np.testing.assert_array_equal(
            runs[label]["solver"]["q_output_cartesian"].values, q
        )


def test_beam_analysis_unchanged_without_dbs(runs):
    """The beam group without DBS is the beam analysis of the full run, less
    the variables the DBS analysis adds"""
    beam_only = runs["beam"]["beam"].to_dataset()
    full = runs["both"]["beam"].to_dataset()
    for name in beam_only.data_vars:
        np.testing.assert_array_equal(beam_only[name].values, full[name].values)


@pytest.mark.parametrize("label", ["beam", "none"])
def test_partial_trees_read_back(runs, label):
    groups = from_output_tree(runs[label])
    assert ("analysis" in groups) == (label == "beam")
    assert "dbs_analysis" not in groups
    to_v3_output(runs[label])
