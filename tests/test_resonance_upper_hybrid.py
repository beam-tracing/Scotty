"""An X-mode ray that heads into the upper hybrid layer, and `upper_hybrid_flag`.

The field is uniform, B = -B0 Z_hat with omega_ce / omega = 0.6, psi = 1 - X / L, and
the density is linear, so Omega_pe^2 / Omega^2 = X / L (as in
test_resonance_uniform_field.py). From the low-density side, the X-mode is reflected at
its R cut-off (X / L = 1 - 0.6 = 0.4) before the upper hybrid layer (X / L = 1 - 0.6^2 =
0.64), so the ray is launched inside the plasma, at X / L = 1, between the upper hybrid
layer and the L cut-off (X / L = 1.6), and heads down the density gradient. Its
refractive index then grows without bound as it nears the layer.

The ray only approaches the layer asymptotically: K_par is conserved while |K| grows,
so K turns perpendicular to B. Without a margin, the solver reached the layer only when
a step overshot it (at 30 deg to the perpendicular and rtol = 1e-4, but not at 45 deg),
and with K perpendicular to B it stalled about 3.6e-7 short of omega_UH / omega = 1 and
never returned. At the default len_tau, the last output points were off the dispersion
relation, with |H| up to 1.4 (Bakso's review of #184, 12 Nov 2026). So the ray stops
`_UPPER_HYBRID_MARGIN` = 1e-2 short of the layer, in omega_UH / omega. Measured at the
default tolerances and len_tau (12 Nov 2026), max |H| is then 3e-4 to 2.8e-3 at 0 to 45
deg (1.2e-2 at 45 deg with a 5e-3 margin, and the solver raised from the H check at 3e-3).
"""

import importlib

import numpy as np
import pytest
from scipy import constants
from scipy.optimize import brentq

from scotty.beam_me_up import beam_me_up
from scotty.geometry import InterpolatedField_Cartesian
from scotty.hamiltonian import Hamiltonian
from scotty.profile_fit import LinearFit
from scotty.ray_solver import _UPPER_HYBRID_MARGIN, _find_UH_ratio, ray_tracing

K0 = 2200.0
OMEGA = K0 * constants.c
LAUNCH_FREQ_GHZ = OMEGA / (2 * np.pi) / 1e9
L = 1600 / K0
GYRO = 0.6
N_CUTOFF = constants.epsilon_0 * constants.m_e * OMEGA**2 / constants.e**2 / 1e19
B0 = GYRO * OMEGA * constants.m_e / constants.e
X = np.linspace(-0.1 * L, 2.0 * L, 22)
Y = np.linspace(-1.0 * L, 2.0 * L, 13)
Z = np.linspace(-0.1 * L, 0.1 * L, 7)
X_GRID, _, _ = np.meshgrid(X, Y, Z, indexing="ij")
LAUNCH = np.array([L, 0.0, 0.0])  # X / L = 1
ANGLES = [0.0, 10.0, 20.0, 30.0, 45.0]


def K_hat(angle_deg=30.0):
    """Along -X and +Y, at `angle_deg` from the perpendicular towards B"""
    a, b = np.deg2rad(30.0), np.deg2rad(angle_deg)
    return np.array([-np.cos(a) * np.cos(b), np.sin(a) * np.cos(b), np.sin(b)])


def field():
    zero = np.zeros_like(X_GRID)
    return InterpolatedField_Cartesian(X, Y, Z, zero, zero, zero - B0, 1 - X_GRID / L)


def hamiltonian(mode_flag=-1):
    return Hamiltonian(
        launch_angular_freq=OMEGA,
        mode_flag=mode_flag,
        deltas=np.array([1e-4, 1e-4, 1e-4, 0.1, 0.1, 0.1]),
        field=field(),
        density_fit=LinearFit(1.0, N_CUTOFF),
    )


def X_mode_K(angle_deg=30.0):
    """K along `K_hat` on the X-mode's dispersion relation at the launch point
    (mode_flag = -1 is the X-mode here; +1 has no root along these directions)"""
    h = hamiltonian(-1)
    direction = K_hat(angle_deg)
    K = brentq(lambda k: h(LAUNCH, k * direction), 0.05 * K0, 5 * K0)
    return K * direction


def run(output_path, upper_hybrid_flag, angle_deg=30.0, **solver):
    """The ray only, with the default tolerances and len_tau unless given"""
    return beam_me_up(
        geometry="cartesian",
        poloidal_launch_angle_Torbeam=0.0,
        toroidal_launch_angle_Torbeam=0.0,
        launch_freq_GHz=LAUNCH_FREQ_GHZ,
        launch_beam_width=0.01,
        launch_beam_curvature=0.0,
        launch_position=LAUNCH,
        mode_flag=-1,
        launch_flag="plasma",
        Psi_BC_flag=None,
        plasmaLaunch_K_cartesian=X_mode_K(angle_deg),
        # Needed by the launch's checks, though only the ray is traced
        plasmaLaunch_Psi_3D_lab_cartesian=(1.0 + 1.0j) * np.eye(3),
        find_B_method=field(),
        density_fit_method=LinearFit(1.0, N_CUTOFF),
        poloidal_flux_zero_density=1.0,
        upper_hybrid_flag=upper_hybrid_flag,
        ray_tracing_flag=True,
        delta_X=1e-4,
        delta_Y=1e-4,
        delta_Z=1e-4,
        output_path=output_path,
        console_log_level="ERROR",
        **solver,
    )


def UH_ratio_and_H(dt):
    beam = dt["beam"]
    UH = np.hypot(
        beam["normalised_omega_ce"].values, beam["normalised_omega_pe"].values
    )
    return UH, np.abs(beam["H_Booker"].values)


def test_launch_is_X_mode_between_the_layer_and_the_L_cutoff():
    for angle in ANGLES:
        assert np.linalg.norm(X_mode_K(angle)) == pytest.approx(K0, rel=1e-5)
    UH = _find_UH_ratio(LAUNCH, OMEGA, field(), hamiltonian())
    assert UH == pytest.approx(np.sqrt(GYRO**2 + 1), rel=1e-5)


@pytest.mark.timeout(60)
@pytest.mark.parametrize("angle", ANGLES)
def test_X_mode_stops_short_of_the_upper_hybrid_layer(tmp_path, angle):
    """The default flag, tolerances and len_tau: the output ends 1e-2 short of the
    layer, on the dispersion relation. At 0 deg (K perpendicular to B) the run
    used not to return"""
    dt = run(tmp_path, "stop", angle)
    UH, H = UH_ratio_and_H(dt)
    # Measured 1.01e-2 to 1.02e-2: the last output step before the event
    assert _UPPER_HYBRID_MARGIN < UH[-1] - 1 < _UPPER_HYBRID_MARGIN + 1e-3
    assert np.all(UH - 1 > _UPPER_HYBRID_MARGIN)
    # |K| grows towards the layer: 3.5 to 4.2 K0 at the last point
    K = np.linalg.norm(dt["solver"]["K_output_cartesian"].values, axis=1)
    assert K[-1] > 3 * K0
    assert len(H) == 4001
    # Measured 3.2e-4 (0 deg) to 2.8e-3 (45 deg)
    assert H.max() < 1e-2


@pytest.mark.timeout(60)
@pytest.mark.parametrize("angle", ANGLES)
def test_X_mode_stops_short_of_the_upper_hybrid_layer_at_v3_tolerances(tmp_path, angle):
    """At v3's rtol = 1e-3 and atol = 1e-6, |H| is already up to 5e-3 far from the
    layer, and the ray stops without raising from the H check (at 0.1)"""
    dt = run(tmp_path, "stop", angle, rtol=1e-3, atol=1e-6)
    UH, H = UH_ratio_and_H(dt)
    assert _UPPER_HYBRID_MARGIN < UH[-1] - 1 < _UPPER_HYBRID_MARGIN + 1e-3
    # Measured 3.3e-3 (10 deg) to 3.1e-2 (45 deg)
    assert H.max() < 5e-2


@pytest.mark.timeout(60)
def test_X_mode_raises_short_of_the_upper_hybrid_layer(tmp_path):
    with pytest.raises(
        RuntimeError, match=f"came within {_UPPER_HYBRID_MARGIN} of the upper hybrid"
    ) as error:
        run(tmp_path, "raise")
    message = str(error.value)
    assert '`upper_hybrid_flag="stop"`' in message
    assert "can't be traced through the layer" in message
    # omega_UH / omega = 1 + margin is at X / L = (1 + margin)^2 - 0.6^2 (to the
    # interpolation of psi)
    q_X = float(message.split("q = [")[1].split()[0])
    assert q_X / L == pytest.approx((1 + _UPPER_HYBRID_MARGIN) ** 2 - GYRO**2, abs=1e-4)


class _ReachedTheSolver(Exception):
    pass


def _start(monkeypatch, X_mode, upper_hybrid_flag, offset):
    """Run `ray_tracing`'s start checks for a ray starting where omega_UH / omega
    = 1 + `offset`, stopping where the solver would start"""
    h = hamiltonian()
    f = h.field
    X_start = brentq(
        lambda x: _find_UH_ratio(np.array([x, 0.0, 0.0]), OMEGA, f, h) - 1 - offset,
        0.6 * L,
        0.7 * L,
        xtol=1e-14,
    )
    q = np.array([X_start, 0.0, 0.0])
    assert _find_UH_ratio(q, OMEGA, f, h) - 1 == pytest.approx(offset, abs=1e-10)

    def solver(*args, **kwargs):
        raise _ReachedTheSolver

    monkeypatch.setattr(
        importlib.import_module("scotty.ray_solver"), "solve_ivp", solver
    )
    ray_tracing(
        q_initial=q,
        K_initial=K0 * K_hat(),
        poloidal_flux_enter=1.0,
        hamiltonian=h,
        ray_tracing_flag=True,
        rtol=1e-4,
        atol=1e-5,
        len_tau=10,
        upper_hybrid_flag=upper_hybrid_flag,
        X_mode=X_mode,
    )


@pytest.mark.parametrize("offset", [0.0, 0.5, -0.5, 0.99, -0.99])
@pytest.mark.parametrize("upper_hybrid_flag", ["stop", "raise"])
def test_X_mode_starting_within_the_margin_raises(
    monkeypatch, upper_hybrid_flag, offset
):
    with pytest.raises(
        RuntimeError, match=f"starts within {_UPPER_HYBRID_MARGIN} of the upper hybrid"
    ):
        _start(monkeypatch, True, upper_hybrid_flag, offset * _UPPER_HYBRID_MARGIN)


@pytest.mark.parametrize(
    ("X_mode", "offset"),
    [(True, 1.5), (True, -1.5), (False, 0.0), (False, 0.5)],
)
@pytest.mark.parametrize("upper_hybrid_flag", ["stop", "raise"])
def test_starting_outside_the_margin_allowed(
    monkeypatch, X_mode, offset, upper_hybrid_flag
):
    """Outside the margin on either side, or for the O-mode, which has no upper
    hybrid resonance"""
    with pytest.raises(_ReachedTheSolver):
        _start(monkeypatch, X_mode, upper_hybrid_flag, offset * _UPPER_HYBRID_MARGIN)


@pytest.mark.parametrize("X_mode", [True, False])
def test_upper_hybrid_flag_pass_is_refused(monkeypatch, X_mode):
    with pytest.raises(ValueError, match="`upper_hybrid_flag` can't be 'pass'"):
        _start(monkeypatch, X_mode, "pass", 2 * _UPPER_HYBRID_MARGIN)
