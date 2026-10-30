"""Plasma entry search along the launch line.

The search length used to be max(q_launch_cartesian), the largest Cartesian
component of the launch point rather than a length, which is too short for
inboard launches (the inboard cases of test_find_plasma_entry_position and
test_resonance). v3 bounded it by the field's (R, Z) grid instead.
"""

from scotty.fun_general import find_q_labframe_cyl_to_cart
from scotty.init_bruv import get_parameters_for_Scotty
from scotty.launch import max_search_distance
from scotty.geometry import create_magnetic_geometry

import numpy as np
import pathlib
import pytest


@pytest.fixture(scope="module")
def field():
    kwargs_dict = get_parameters_for_Scotty("DBS_synthetic")
    kwargs_dict["find_B_method"] = "unit-tests"
    kwargs_dict.pop("interp_smoothing", None)
    kwargs_dict.pop("magnetic_data_path", None)
    return create_magnetic_geometry(
        interp_order_str="quintic",
        interp_order_int=5,
        interp_smoothing=0,
        magnetic_data_path=pathlib.Path("."),
        **kwargs_dict,
    )


@pytest.mark.parametrize(
    "q_launch_cylindrical",
    [
        np.array([2.3, 0.0, -0.1]),  # outboard, near the midplane
        np.array([0.1, 0.0, 0.0]),  # inboard
        np.array([2.3, 1.0, 0.8]),  # off the midplane, at zeta = 1
        np.array([2.3, np.pi, -0.6]),  # at zeta = pi, where X < 0
    ],
    ids=["outboard", "inboard", "off-midplane", "zeta-pi"],
)
def test_search_reaches_the_whole_domain(field, q_launch_cylindrical):
    """Every point of the field's grid, at any toroidal angle, lies within
    max_search_distance of the launch point"""
    q_launch = find_q_labframe_cyl_to_cart(q_launch_cylindrical)
    distance = max_search_distance(field, q_launch)
    R, zeta, Z = np.meshgrid(
        field.R_coord, np.linspace(-np.pi, np.pi, 37), field.Z_coord, indexing="ij"
    )
    points = np.stack([R * np.cos(zeta), R * np.sin(zeta), Z], axis=-1)
    assert np.max(np.linalg.norm(points - q_launch, axis=-1)) <= distance


@pytest.mark.parametrize("scale", [1.0, 1.37, 2.0])
def test_entry_does_not_depend_on_the_search_grid(field, monkeypatch, scale):
    """The entry point is refined on the flux itself, so lengthening the
    search (which changes the sampling grid) must not move it. It must lie
    inside the plasma, by no more than rounding: this is where the density
    is still zero and the vacuum K satisfies H = 0"""
    from scotty import launch

    kwargs = dict(
        poloidal_launch_angle_deg_Torbeam=-1.0,
        toroidal_launch_angle_deg_Torbeam=10.0,
        q_launch=np.array([2.3, 0.0, -0.1]),
        launch_flag="vacuum",
        field=field,
        poloidal_flux_enter=1.0,
    )
    reference = launch.find_plasma_entry_position(**kwargs)
    original = launch.max_search_distance
    monkeypatch.setattr(
        launch, "max_search_distance", lambda f, q: scale * original(f, q)
    )
    entry = launch.find_plasma_entry_position(**kwargs)
    np.testing.assert_allclose(entry, reference, rtol=0, atol=1e-9)
    R, zeta, Z = entry
    flux = field.polflux(R, zeta, Z)
    assert flux <= 1.0
    assert flux > 1.0 - 1e-9
