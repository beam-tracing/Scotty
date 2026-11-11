"""The ray solver's resonance events (beam-tracing/Scotty#151), checked directly.

The ported #151 trajectory tests never reach the upper-hybrid layer, so they pass with or without
its event. Here each event is evaluated at points on either side of the resonances on a line
through the plasma, and must change sign across its own layer and no other: the first harmonic
(the fundamental) and the second harmonic of the electron cyclotron frequency in both modes, and
the upper-hybrid layer in X-mode only. A layer whose flag is "pass" has no event.
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
from scotty.ray_solver import _tau_short_of_first_harmonic, make_solver_events

R_AXIS, A, B_T_AXIS, B_P_A = 1.5, 0.5, 1.0, 0.1
FIELD = CircularFluxSurfaceField(B_T_AXIS, R_AXIS, A, B_P_A)
DENSITY = QuadraticFit(1.0, 5.0)  # 5e19 m^-3 on axis
OMEGA = freq_GHz_to_angular_frequency(50.0)
Z = 0.0
RESONANCES = ("first_harmonic", "second_harmonic", "upper_hybrid")


@pytest.fixture(scope="module", autouse=True)
def trace_level(tmp_path_factory):
    config_logger("warning", None, tmp_path_factory.mktemp("log"), "")


def events(mode_flag, temperature_fit=None, X_mode=None, omega=OMEGA, **flags):
    """The resonance events for the Hamiltonian with `mode_flag`, as functions of
    R on the midplane. Unless given, `X_mode` is True for mode_flag = -1, the
    X-mode in this geometry"""
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
    all_events = make_solver_events(1.0, omega, FIELD, X_mode=X_mode, **flags)

    def at_R(event):
        return lambda R: event(0.0, [R, 0.0, Z, 0.0, 0.0, 0.0], hamiltonian)

    return {
        name: at_R(event) for name, event in all_events.items() if name in RESONANCES
    }


def changes_sign(f, R_layer):
    return np.sign(f(R_layer - 1e-3)) != np.sign(f(R_layer + 1e-3))


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


@pytest.mark.parametrize("mode_flag", [1, -1])
def test_events_present_by_default(mode_flag):
    """The first and second harmonics for both modes, the upper hybrid for the
    X-mode only"""
    expected = {"first_harmonic", "second_harmonic"} | (
        {"upper_hybrid"} if mode_flag == -1 else set()
    )
    assert set(events(mode_flag)) == expected


@pytest.mark.parametrize(
    ("mode_flag", "name", "R_layer"),
    [
        pytest.param(1, "second_harmonic", SECOND, id="O-second-harmonic"),
        pytest.param(-1, "second_harmonic", SECOND, id="X-second-harmonic"),
        pytest.param(-1, "upper_hybrid", UH, id="X-upper-hybrid"),
    ],
)
def test_each_event_changes_sign_across_its_own_layer_only(mode_flag, name, R_layer):
    """On the 50 GHz outboard line, the layer's event changes sign across it and
    not across the other layer, and no other event changes sign across it (the
    fundamental isn't on this line)"""
    fs = events(mode_flag)
    other_R = UH if R_layer == SECOND else SECOND
    assert changes_sign(fs[name], R_layer)
    assert not changes_sign(fs[name], other_R)
    for other_name, f in fs.items():
        if other_name != name:
            assert not changes_sign(f, R_layer), other_name


@pytest.mark.parametrize("mode_flag", [1, -1])
def test_first_harmonic_event(mode_flag):
    """At 30 GHz the fundamental is inside the plasma, on the inboard side. Only
    the first-harmonic event changes sign across it"""
    omega = freq_GHz_to_angular_frequency(30.0)
    fs = events(mode_flag, omega=omega)
    f = fs["first_harmonic"]
    R_layer = brentq(f, R_AXIS - A + 1e-3, R_AXIS - 1e-3)
    B = float(FIELD.magnitude(R_layer, 0.0, Z))
    assert float(find_normalised_gyro_ang_freq(omega, B)) == pytest.approx(1.0)
    assert changes_sign(f, R_layer)
    for name, other in fs.items():
        if name != "first_harmonic":
            assert not changes_sign(other, R_layer), name


@pytest.mark.parametrize(
    ("mode_flag", "X_mode", "must_exist"),
    [
        pytest.param(1, True, True, id="X-mode-as-mode_flag+1"),
        pytest.param(-1, False, False, id="O-mode-as-mode_flag-1"),
    ],
)
def test_upper_hybrid_follows_the_X_mode_not_mode_flag(mode_flag, X_mode, must_exist):
    """Which mode_flag is the X-mode depends on the plasma and K, so the
    upper-hybrid layer is detected for the X-mode whatever its mode_flag"""
    fs = events(mode_flag, X_mode=X_mode)
    assert ("upper_hybrid" in fs) == must_exist
    if must_exist:
        assert changes_sign(fs["upper_hybrid"], UH)
    # No other event changes sign at the upper-hybrid layer
    for name, f in fs.items():
        if name != "upper_hybrid":
            assert not changes_sign(f, UH), name


@pytest.mark.parametrize(
    ("flags", "absent"),
    [
        ({"second_harmonic_flag": "pass"}, "second_harmonic"),
        ({"upper_hybrid_flag": "pass"}, "upper_hybrid"),
    ],
)
@pytest.mark.parametrize("first_harmonic_flag", ["stop", "raise"])
def test_pass_has_no_event(flags, absent, first_harmonic_flag):
    fs = events(-1, first_harmonic_flag=first_harmonic_flag, **flags)
    assert absent not in fs
    assert "first_harmonic" in fs
    assert set(fs) == set(RESONANCES) - {absent}


def test_first_harmonic_cant_be_passed():
    with pytest.raises(ValueError, match="first_harmonic_flag"):
        events(1, first_harmonic_flag="pass")


def test_two_layers_crossed_in_one_step_are_both_seen():
    """A solver step from before both the second harmonic and the upper hybrid
    layer to after both: each event changes sign, whereas their product (the
    single event used before) has the same sign at both ends and would miss both"""
    before, after = min(SECOND, UH) - 1e-3, max(SECOND, UH) + 1e-3
    fs = events(-1)
    second, upper_hybrid = fs["second_harmonic"], fs["upper_hybrid"]
    assert np.sign(second(before)) != np.sign(second(after))
    assert np.sign(upper_hybrid(before)) != np.sign(upper_hybrid(after))
    assert np.sign(second(before) * upper_hybrid(before)) == np.sign(
        second(after) * upper_hybrid(after)
    )


def test_relativistic_correction_moves_the_layer():
    """With a temperature, the relativistic mass increase lowers the cyclotron frequency, so the
    second-harmonic layer moves to stronger field (smaller R on the outboard side)"""
    hot = QuadraticFit(1.0, 20.0)  # 20 keV on axis
    f_cold = events(1)["second_harmonic"]
    f_hot = events(1, temperature_fit=hot)["second_harmonic"]
    R_cold = brentq(f_cold, SECOND - 0.05, SECOND + 0.05)
    R_hot = brentq(f_hot, R_AXIS + 1e-3, SECOND + 0.05)
    assert np.isclose(R_cold, SECOND, atol=1e-6)
    assert R_hot < R_cold - 1e-3


def test_output_stop_is_the_last_approach_to_the_first_harmonic():
    """A ray that comes within the margin, turns away and comes back has several
    crossings of the margin. The output must stop at the last one (tau = 2.5 here),
    not at an earlier near-approach (0.5 or 1.5)"""
    distance = lambda tau: 1e-3 * np.cos(np.pi * tau)
    steps = np.linspace(0.0, 3.0, 31)
    assert _tau_short_of_first_harmonic(steps, 3.0, distance) == pytest.approx(2.5)


def test_output_stop_leaves_other_ends_alone():
    """Unchanged if the ray ends outside the margin (another layer, or the plasma
    edge), or if it is within the margin at every step"""
    steps = np.linspace(0.0, 1.0, 11)
    assert _tau_short_of_first_harmonic(steps, 1.0, lambda tau: 0.1) == 1.0
    assert _tau_short_of_first_harmonic(steps, 1.0, lambda tau: -0.1) == 1.0
