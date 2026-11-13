"""`launch_beam_width` and `launch_beam_curvature`: a number is a circular beam, as before.
A 2-tuple and a 3-tuple are reserved for elliptical and astigmatic beams, until the launch
beam frame and the meaning of their elements are decided, so they raise NotImplementedError.
Anything else raises ValueError."""

import numpy as np
import pytest

from scotty.beam_me_up import beam_me_up
from scotty.checks import _circular_launch_beam
from scotty.init_bruv import get_parameters_for_Scotty


@pytest.mark.parametrize("value", [0.04, -0.25, np.float64(0.04), np.array(0.04), 1])
def test_number_is_circular(value):
    assert _circular_launch_beam("launch_beam_width", value) == float(value)


@pytest.mark.parametrize(
    "value",
    [(0.03, 0.05), (0.03, 0.05, 0.5), [0.03, 0.05], np.array([0.03, 0.05, 0.5])],
)
def test_tuples_are_reserved(value):
    with pytest.raises(NotImplementedError, match="reserved"):
        _circular_launch_beam("launch_beam_width", value)


@pytest.mark.parametrize("value", [(0.04,), (1, 2, 3, 4), np.eye(2), "0.04", None])
def test_other_forms_raise(value):
    with pytest.raises(ValueError, match="must be a number"):
        _circular_launch_beam("launch_beam_width", value)


@pytest.mark.parametrize("name", ["launch_beam_width", "launch_beam_curvature"])
def test_beam_me_up_refuses_a_tuple_before_running(tmp_path, name):
    parameters = get_parameters_for_Scotty("DBS_synthetic")
    parameters.update(
        launch_flag="vacuum_analytic",
        Psi_BC_flag="continuous",
        console_log_level="WARNING",
        output_path=tmp_path,
    )
    parameters[name] = (parameters[name], parameters[name])
    with pytest.raises(NotImplementedError, match=name):
        beam_me_up(**parameters)
    assert not any(tmp_path.iterdir())  # stopped before writing anything
