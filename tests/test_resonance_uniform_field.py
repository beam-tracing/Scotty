"""A ray in a uniform field exactly at a resonance, with `resonance_flag=False`.

The field is B = -B0 Z_hat with psi = 1 - X / L, and the density is linear, so
Omega_pe^2 / Omega^2 = X / L. For an O-mode with K perpendicular to B, the ray is
then known exactly and doesn't depend on B0. It enters at 45 deg, so it turns at
X / L = 0.5, before the X-mode's upper hybrid layer at X / L = 1 - Y^2 = 0.75.

Exactly at the second harmonic, the old resonance event was zero all along the ray,
so the ray stopped early with no error. Exactly at the fundamental, the solver
never returned.
"""

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy import constants

from scotty.beam_me_up import beam_me_up
from scotty.geometry import InterpolatedField_Cartesian
from scotty.profile_fit import LinearFit

K0 = 2200.0
OMEGA = K0 * constants.c
LAUNCH_FREQ_GHZ = OMEGA / (2 * np.pi) / 1e9
L = 1600 / K0
A0 = np.deg2rad(45.0)
N_CUTOFF = constants.epsilon_0 * constants.m_e * OMEGA**2 / constants.e**2 / 1e19
TAU_LAUNCH = 1e-3
TAU_EXIT = 2 * K0 * L * np.cos(A0)


def analytic_K_X(tau):
    return K0 * np.cos(A0) - tau / L


def run(gyro_ratio, resonance_flag, output_path):
    B0 = gyro_ratio * OMEGA * constants.m_e / constants.e
    X = np.linspace(-0.1 * L, 1.1 * L, 13)
    Y = np.linspace(-0.1 * L, 2.1 * L, 13)
    Z = np.linspace(-0.1 * L, 0.1 * L, 7)
    X_grid, _, _ = np.meshgrid(X, Y, Z, indexing="ij")
    zero = np.zeros_like(X_grid)
    K_X = analytic_K_X(TAU_LAUNCH)
    K_Y = K0 * np.sin(A0)
    return beam_me_up(
        geometry="cartesian",
        poloidal_launch_angle_Torbeam=0.0,
        toroidal_launch_angle_Torbeam=0.0,
        launch_freq_GHz=LAUNCH_FREQ_GHZ,
        launch_beam_width=0.01,
        launch_beam_curvature=0.0,
        launch_position=np.array(
            [
                L * (np.cos(A0) ** 2 - (K_X / K0) ** 2),
                2 * np.sin(A0) * TAU_LAUNCH / K0,
                0.0,
            ]
        ),
        mode_flag="O",
        launch_flag="plasma",
        Psi_BC_flag=None,
        plasmaLaunch_K_cartesian=np.array([K_X, K_Y, 0.0]),
        # Needed by the launch's checks, though only the ray is traced
        plasmaLaunch_Psi_3D_lab_cartesian=(1.0 + 1.0j) * np.eye(3),
        find_B_method=InterpolatedField_Cartesian(
            X, Y, Z, zero, zero, zero - B0, 1 - X_grid / L
        ),
        density_fit_method=LinearFit(1.0, N_CUTOFF),
        poloidal_flux_zero_density=1.0,
        resonance_flag=resonance_flag,
        ray_tracing_flag=True,
        delta_X=1e-3,
        delta_Y=1e-3,
        delta_Z=1e-3,
        len_tau=50,
        rtol=1e-8,
        atol=1e-9,
        figure_flag=False,
        output_path=output_path,
        console_log_level="WARNING",
    )


def test_second_harmonic_doesnt_stop_the_ray(tmp_path):
    so = run(0.5, False, tmp_path)["solver_output"]
    tau = so["tau"].values
    step = tau[1] - tau[0]
    assert abs(tau[-1] + step + TAU_LAUNCH - TAU_EXIT) < 2 * step
    assert_allclose(
        so["K_output_cartesian"].values[:, 0],
        analytic_K_X(tau + TAU_LAUNCH),
        rtol=0,
        atol=1e-5 * K0,
    )


@pytest.mark.parametrize("resonance_flag", [True, False])
def test_fundamental_raises_rather_than_hangs(tmp_path, resonance_flag):
    with pytest.raises(RuntimeError, match="starts on the fundamental"):
        run(1.0, resonance_flag, tmp_path)
