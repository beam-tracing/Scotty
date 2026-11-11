"""The ray solver's resonance event (beam-tracing/Scotty#151), checked directly.

The ported #151 trajectory tests never reach the upper-hybrid layer, so they pass with or without
its term in the event. Here the event is evaluated at points on either side of each resonance on a
line through the plasma, and must change sign across exactly the layers it is meant to detect:
the cyclotron fundamental and second harmonic in both modes, and the upper-hybrid layer in X-mode
only. With resonance_flag 1 or 0 it must change sign at the fundamental only.
"""

import numpy as np
import pytest
from scipy.optimize import brentq

from scotty.fun_general import (
    find_normalised_gyro_ang_freq,
    find_normalised_plasma_ang_freq,
    freq_GHz_to_angular_frequency,
)
from scotty.checks import check_resonance_flag
from scotty.geometry import CircularFluxSurfaceField
from scotty.hamiltonian import Hamiltonian
from scotty.logger import config_logger
from scotty.profile_fit import QuadraticFit
from scotty.ray_solver import _tau_short_of_fundamental, make_solver_events

R_AXIS, A, B_T_AXIS, B_P_A = 1.5, 0.5, 1.0, 0.1
FIELD = CircularFluxSurfaceField(B_T_AXIS, R_AXIS, A, B_P_A)
DENSITY = QuadraticFit(1.0, 5.0)  # 5e19 m^-3 on axis
OMEGA = freq_GHz_to_angular_frequency(50.0)
Z = 0.0


@pytest.fixture(scope="module", autouse=True)
def trace_level(tmp_path_factory):
    config_logger("warning", None, tmp_path_factory.mktemp("log"), "")


def event(mode_flag, resonance_flag, temperature_fit=None, X_mode=None, omega=OMEGA):
    """The resonance event for the Hamiltonian with `mode_flag`. Unless given,
    `X_mode` is True for mode_flag = -1, the X-mode in this geometry"""
    hamiltonian = Hamiltonian(
        launch_angular_freq=omega,
        mode_flag=mode_flag,
        deltas=np.array([1e-4, 1e-4, 0.1, 0.1, 0.1]),
        field=FIELD,
        density_fit=DENSITY,
        temperature_fit=temperature_fit,
    )
    if X_mode is None:
        X_mode = mode_flag == -1
    events = make_solver_events(1.0, omega, FIELD, resonance_flag, X_mode=X_mode)
    cross = events["cross_resonance"]
    return lambda R: cross(0.0, [R, 0.0, Z, 0.0, 0.0, 0.0], hamiltonian)


def normalised(R):
    B = float(FIELD.magnitude(R, 0.0, Z))
    n_e = float(DENSITY(FIELD.polflux(R, 0.0, Z)))
    gyro = float(find_normalised_gyro_ang_freq(OMEGA, B))
    plasma = float(find_normalised_plasma_ang_freq(OMEGA, n_e))
    return gyro, plasma


def layer(f):
    """R of the layer where f(R) crosses zero, between the axis and the outboard edge"""
    return brentq(f, R_AXIS + 1e-3, R_AXIS + A - 1e-3)


# Layers along the outboard midplane at 50 GHz, where f_ce runs from 28 GHz (axis) to 21 GHz (edge):
# the second harmonic (gyro = 0.5) is inside the plasma, and so is the upper-hybrid layer
# (gyro^2 + plasma^2 = 1), where f_UH falls from ~69 GHz to 21 GHz. The fundamental (f_ce = 50 GHz)
# is not reached anywhere in this plasma
UH = layer(lambda R: np.hypot(*normalised(R)) - 1)
SECOND = layer(lambda R: normalised(R)[0] - 0.5)


def test_layers_are_separated():
    assert abs(UH - SECOND) > 0.02


@pytest.mark.parametrize(
    ("mode_flag", "R_layer", "must_change"),
    [
        pytest.param(1, SECOND, True, id="O-second-harmonic"),
        pytest.param(-1, SECOND, True, id="X-second-harmonic"),
        pytest.param(-1, UH, True, id="X-upper-hybrid"),
        pytest.param(1, UH, False, id="O-no-upper-hybrid"),
    ],
)
def test_event_changes_sign_across_its_layers(mode_flag, R_layer, must_change):
    f = event(mode_flag, resonance_flag=2)
    before, after = f(R_layer - 1e-3), f(R_layer + 1e-3)
    assert (np.sign(before) != np.sign(after)) == must_change


@pytest.mark.parametrize(
    ("mode_flag", "X_mode", "must_change"),
    [
        pytest.param(1, True, True, id="X-mode-as-mode_flag+1"),
        pytest.param(-1, False, False, id="O-mode-as-mode_flag-1"),
    ],
)
def test_upper_hybrid_follows_the_X_mode_not_mode_flag(mode_flag, X_mode, must_change):
    """Which mode_flag is the X-mode depends on the plasma and K, so the
    upper-hybrid layer is detected for the X-mode whatever its mode_flag"""
    f = event(mode_flag, resonance_flag=2, X_mode=X_mode)
    before, after = f(UH - 1e-3), f(UH + 1e-3)
    assert (np.sign(before) != np.sign(after)) == must_change


@pytest.mark.parametrize("resonance_flag", [1, 0])
@pytest.mark.parametrize("mode_flag", [1, -1])
def test_resonance_flag_1_and_0_ignore_the_other_layers(mode_flag, resonance_flag):
    """The second harmonic and the upper-hybrid layer are on this line, and the
    fundamental isn't. The event stays nonzero with one sign, including exactly
    at the second harmonic, where the old event was zero"""
    f = event(mode_flag, resonance_flag=resonance_flag)
    R = np.linspace(R_AXIS + 1e-3, R_AXIS + A - 1e-3, 401)
    values = np.array([f(r) for r in [*R, SECOND, UH]])
    assert np.all(values < 0)


@pytest.mark.parametrize("resonance_flag", [1, 0])
@pytest.mark.parametrize("mode_flag", [1, -1])
def test_resonance_flag_1_and_0_detect_the_fundamental(mode_flag, resonance_flag):
    """At 30 GHz the fundamental is inside the plasma, on the inboard side"""
    omega = freq_GHz_to_angular_frequency(30.0)
    f = event(mode_flag, resonance_flag=resonance_flag, omega=omega)
    R_layer = brentq(f, R_AXIS - A + 1e-3, R_AXIS - 1e-3)
    B = float(FIELD.magnitude(R_layer, 0.0, Z))
    assert float(find_normalised_gyro_ang_freq(omega, B)) == pytest.approx(1.0)
    assert np.sign(f(R_layer - 1e-3)) != np.sign(f(R_layer + 1e-3))


def test_relativistic_correction_moves_the_layer():
    """With a temperature, the relativistic mass increase lowers the cyclotron frequency, so the
    second-harmonic layer moves to stronger field (smaller R on the outboard side)"""
    hot = QuadraticFit(1.0, 20.0)  # 20 keV on axis
    f_cold = event(1, resonance_flag=2)
    f_hot = event(1, resonance_flag=2, temperature_fit=hot)
    R_cold = brentq(f_cold, SECOND - 0.05, SECOND + 0.05)
    R_hot = brentq(f_hot, R_AXIS + 1e-3, SECOND + 0.05)
    assert np.isclose(R_cold, SECOND, atol=1e-6)
    assert R_hot < R_cold - 1e-3


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (2, 2),
        (1, 1),
        (0, 0),
        (True, 2),
        (False, 0),
        (np.int64(1), 1),
        (np.bool_(True), 2),
    ],
)
def test_check_resonance_flag(value, expected):
    """True and False are 2 and 0, not 1 and 0, although True == 1 in Python"""
    result = check_resonance_flag(value)
    assert result == expected and type(result) is int


@pytest.mark.parametrize("value", [3, -1, 1.0, "yes", None])
def test_check_resonance_flag_rejects_other_values(value):
    with pytest.raises(ValueError, match="must be 0, 1 or 2"):
        check_resonance_flag(value)


def test_output_stop_is_the_last_approach_to_the_fundamental():
    """A ray that comes within the margin, turns away and comes back has several
    crossings of the margin. The output must stop at the last one (tau = 2.5 here),
    not at an earlier near-approach (0.5 or 1.5)"""
    distance = lambda tau: 1e-3 * np.cos(np.pi * tau)
    steps = np.linspace(0.0, 3.0, 31)
    assert _tau_short_of_fundamental(steps, 3.0, distance) == pytest.approx(2.5)


def test_output_stop_leaves_other_ends_alone():
    """Unchanged if the ray ends outside the margin (another layer, or the plasma
    edge), or if it is within the margin at every step"""
    steps = np.linspace(0.0, 1.0, 11)
    assert _tau_short_of_fundamental(steps, 1.0, lambda tau: 0.1) == 1.0
    assert _tau_short_of_fundamental(steps, 1.0, lambda tau: -0.1) == 1.0
