"""The ray solver's resonance event (beam-tracing/Scotty#151), checked directly.

The ported #151 trajectory tests never reach the upper-hybrid layer, so they pass with or without
its term in the event. Here the event is evaluated at points on either side of each resonance on a
line through the plasma, and must change sign across exactly the layers it is meant to detect:
the cyclotron fundamental and second harmonic in both modes, and the upper-hybrid layer in X-mode
only. With resonance_flag=False it must never change sign.
"""

import numpy as np
import pytest
from scipy.optimize import brentq

from scotty.fun_general import (
    find_normalised_gyro_ang_freq,
    find_normalised_plasma_ang_freq,
    freq_GHz_to_angular_frequency,
)
from scotty.geometry import CircularFluxSurfaceField
from scotty.hamiltonian import Hamiltonian
from scotty.logger import config_logger
from scotty.profile_fit import QuadraticFit
from scotty.ray_solver import make_solver_events

R_AXIS, A, B_T_AXIS, B_P_A = 1.5, 0.5, 1.0, 0.1
FIELD = CircularFluxSurfaceField(B_T_AXIS, R_AXIS, A, B_P_A)
DENSITY = QuadraticFit(1.0, 5.0)  # 5e19 m^-3 on axis
OMEGA = freq_GHz_to_angular_frequency(50.0)
Z = 0.0


@pytest.fixture(scope="module", autouse=True)
def trace_level(tmp_path_factory):
    config_logger("warning", None, tmp_path_factory.mktemp("log"), "")


def event(mode_flag, resonance_flag, temperature_fit=None):
    hamiltonian = Hamiltonian(
        launch_angular_freq=OMEGA,
        mode_flag=mode_flag,
        deltas=np.array([1e-4, 1e-4, 0.1, 0.1, 0.1]),
        field=FIELD,
        density_fit=DENSITY,
        temperature_fit=temperature_fit,
    )
    events = make_solver_events(1.0, OMEGA, FIELD, resonance_flag)
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
    f = event(mode_flag, resonance_flag=True)
    before, after = f(R_layer - 1e-3), f(R_layer + 1e-3)
    assert (np.sign(before) != np.sign(after)) == must_change


@pytest.mark.parametrize("mode_flag", [1, -1])
def test_resonance_flag_false_never_changes_sign(mode_flag):
    f = event(mode_flag, resonance_flag=False)
    R = np.linspace(R_AXIS + 1e-3, R_AXIS + A - 1e-3, 401)
    values = np.array([f(r) for r in R])
    assert np.all(values >= 0)


def test_relativistic_correction_moves_the_layer():
    """With a temperature, the relativistic mass increase lowers the cyclotron frequency, so the
    second-harmonic layer moves to stronger field (smaller R on the outboard side)"""
    hot = QuadraticFit(1.0, 20.0)  # 20 keV on axis
    f_cold = event(1, resonance_flag=True)
    f_hot = event(1, resonance_flag=True, temperature_fit=hot)
    R_cold = brentq(f_cold, SECOND - 0.05, SECOND + 0.05)
    R_hot = brentq(f_hot, R_AXIS + 1e-3, SECOND + 0.05)
    assert np.isclose(R_cold, SECOND, atol=1e-6)
    assert R_hot < R_cold - 1e-3
