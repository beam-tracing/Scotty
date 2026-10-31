import pathlib

import numpy as np
import pytest
from numpy.testing import assert_allclose

from scotty.beam_me_up import beam_me_up
from scotty.geometry import create_magnetic_geometry
from scotty.init_bruv import get_parameters_for_Scotty
from scotty.launch import find_auto_delta_signs


@pytest.fixture(scope="module")
def field():
    # circular flux surfaces: psi increases away from the axis at (R, Z) = (1.5, 0)
    return create_magnetic_geometry(
        geometry="cylindrical",
        find_B_method="analytical",
        interp_order_str="quintic",
        interp_order_int=5,
        interp_smoothing=0,
        magnetic_data_path=pathlib.Path("."),
        B_T_axis=1.0,
        R_axis=1.5,
        minor_radius_a=0.5,
        B_p_a=0.1,
    )


@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize(
    ("q", "expected_signs"),
    [
        pytest.param(np.array([1.9, 0.0, 0.1]), (-1, -1), id="outboard-above"),
        pytest.param(np.array([1.1, 0.0, -0.1]), (1, 1), id="inboard-below"),
    ],
)
def test_steps_point_into_the_plasma_whatever_their_sign(
    field, q, expected_signs, sign
):
    deltas = sign * np.array([1e-5, 2e-5])
    result = find_auto_delta_signs(True, q, deltas, field)
    # down the flux gradient (into the plasma), with the magnitudes unchanged
    assert_allclose(result, np.array(expected_signs) * np.abs(deltas))


def test_no_auto_sign_keeps_the_steps(field):
    deltas = np.array([-1e-5, 1e-5])
    assert_allclose(
        find_auto_delta_signs(False, np.array([1.9, 0.0, 0.1]), deltas, field), deltas
    )


def test_boundary_condition_independent_of_step_sign(tmp_path):
    """The plasma-side Psi from the continuous boundary condition needs the
    gradient of H on the plasma side, so it must not depend on the sign the
    user gives the steps (v3 scripts used negative ones)"""
    psi = {}
    for sign in (1, -1):
        kwargs = get_parameters_for_Scotty("DBS_synthetic")
        kwargs.update(
            len_tau=52, interp_smoothing=0, delta_R=sign * 1e-5, delta_Z=sign * 1e-5
        )
        out = beam_me_up(
            **kwargs, output_path=tmp_path, output_filename_suffix=f"_{sign}"
        )
        psi[sign] = out["solver_output"]["Psi_3D_initial_labframe_cylindrical"].values
    assert_allclose(psi[-1], psi[1], rtol=1e-6)
