"""An X-mode ray that reaches the upper hybrid layer, and `upper_hybrid_flag`.

The field is uniform, B = -B0 Z_hat with omega_ce / omega = 0.6, psi = 1 - X / L, and
the density is linear, so Omega_pe^2 / Omega^2 = X / L (as in
test_resonance_uniform_field.py). From the low-density side, the X-mode is reflected at
its R cut-off (X / L = 1 - 0.6 = 0.4) before the upper hybrid layer (X / L = 1 - 0.6^2 =
0.64), so the ray is launched inside the plasma, at X / L = 1, between the upper hybrid
layer and the L cut-off (X / L = 1.6), and heads down the density gradient. Its
refractive index then grows without bound as it nears the layer.

Whether the solver gets to the layer depends on the case. With K perpendicular to B
the ray stalls just short of it: |K| grows, the steps shrink, and tau stops advancing
(about 3.6e-7 short of omega_UH / omega = 1, with |H| ~ 1e5, measured 12 Nov 2026). With
K at 30 deg to the perpendicular, as here, a solver step at rtol = 1e-4 lands past the
layer, so the event is seen. At other angles and tolerances the solver can instead fail
("Integration step failed"). "pass" isn't tested: the ray goes a little past the layer
and then stalls.
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
from scotty.ray_solver import _ON_LAYER_TOL, _find_UH_ratio, ray_tracing

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
# Along -X and +Y, with 30 deg towards B
_A, _B = np.deg2rad(30.0), np.deg2rad(30.0)
K_HAT = np.array([-np.cos(_A) * np.cos(_B), np.sin(_A) * np.cos(_B), np.sin(_B)])


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


def X_mode_K():
    """K along `K_HAT` on the X-mode's dispersion relation at the launch point
    (mode_flag = -1 is the X-mode here; +1 has no root along this direction)"""
    h = hamiltonian(-1)
    K = brentq(lambda k: h(LAUNCH, k * K_HAT), 0.05 * K0, 5 * K0)
    return K * K_HAT


def run(output_path, upper_hybrid_flag):
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
        plasmaLaunch_K_cartesian=X_mode_K(),
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
        len_tau=200,
        rtol=1e-4,
        atol=1e-5,
        figure_flag=False,
        output_path=output_path,
        console_log_level="ERROR",
    )


def test_launch_is_X_mode_between_the_layer_and_the_L_cutoff():
    assert np.linalg.norm(X_mode_K()) == pytest.approx(K0, rel=1e-5)
    UH = _find_UH_ratio(LAUNCH, OMEGA, field(), hamiltonian())
    assert UH == pytest.approx(np.sqrt(GYRO**2 + 1), rel=1e-5)


@pytest.mark.timeout(60)
def test_X_mode_stops_at_the_upper_hybrid_layer(tmp_path):
    """The default: the output ends at the layer, one output step short of it,
    still on the dispersion relation"""
    dt = run(tmp_path, "stop")
    beam = dt["beam"]
    UH = np.hypot(
        beam["normalised_omega_ce"].values, beam["normalised_omega_pe"].values
    )
    # Measured 5.9e-3, the last output step
    assert 0 < UH[-1] - 1 < 1e-2
    assert np.all(UH > 1)
    # |K| grows towards the layer: 5.0 K0 at the last point
    K = np.linalg.norm(dt["solver"]["K_output_cartesian"].values, axis=1)
    assert K[-1] > 4 * K0
    # Measured 3.9e-3, at the last point
    assert np.abs(beam["H_Booker"].values).max() < 1e-2


@pytest.mark.timeout(60)
def test_X_mode_raises_at_the_upper_hybrid_layer(tmp_path):
    with pytest.raises(
        RuntimeError, match="reached the upper hybrid resonance"
    ) as error:
        run(tmp_path, "raise")
    message = str(error.value)
    assert '`upper_hybrid_flag="stop"`' in message
    assert '`upper_hybrid_flag="pass"`' in message
    # The layer is at X / L = 0.64 (to the interpolation of psi)
    q_X = float(message.split("q = [")[1].split()[0])
    assert q_X / L == pytest.approx(1 - GYRO**2, abs=1e-4)


class _ReachedTheSolver(Exception):
    pass


def _start(monkeypatch, X_mode, upper_hybrid_flag):
    """Run `ray_tracing`'s start checks for a ray starting on the upper hybrid
    layer, stopping where the solver would start"""
    h = hamiltonian()
    f = h.field
    X_layer = brentq(
        lambda x: _find_UH_ratio(np.array([x, 0.0, 0.0]), OMEGA, f, h) - 1,
        0.6 * L,
        0.7 * L,
        xtol=1e-14,
    )
    q = np.array([X_layer, 0.0, 0.0])
    assert abs(_find_UH_ratio(q, OMEGA, f, h) - 1) < _ON_LAYER_TOL

    def solver(*args, **kwargs):
        raise _ReachedTheSolver

    monkeypatch.setattr(
        importlib.import_module("scotty.ray_solver"), "solve_ivp", solver
    )
    ray_tracing(
        q_initial=q,
        K_initial=K0 * K_HAT,
        poloidal_flux_enter=1.0,
        hamiltonian=h,
        ray_tracing_flag=True,
        rtol=1e-4,
        atol=1e-5,
        len_tau=10,
        upper_hybrid_flag=upper_hybrid_flag,
        X_mode=X_mode,
    )


@pytest.mark.parametrize("upper_hybrid_flag", ["stop", "raise"])
def test_X_mode_starting_on_the_upper_hybrid_layer_raises(
    monkeypatch, upper_hybrid_flag
):
    with pytest.raises(RuntimeError, match="starts on the upper hybrid") as error:
        _start(monkeypatch, True, upper_hybrid_flag)
    assert '`upper_hybrid_flag="pass"`' in str(error.value)


@pytest.mark.parametrize(
    ("X_mode", "upper_hybrid_flag"),
    [(True, "pass"), (False, "stop"), (False, "raise")],
)
def test_starting_on_the_upper_hybrid_layer_allowed(
    monkeypatch, X_mode, upper_hybrid_flag
):
    """When passed, or for the O-mode, which has no upper hybrid resonance"""
    with pytest.raises(_ReachedTheSolver):
        _start(monkeypatch, X_mode, upper_hybrid_flag)
