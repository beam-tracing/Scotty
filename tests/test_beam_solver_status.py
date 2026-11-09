"""A failed beam solve raises rather than returning the points before the failure.

With H = K0, the ray moves along q0 at unit speed (q0 = tau) and Psi stays
constant, so the beam reaches tau_leave unless H is undefined on the way. Making
H NaN beyond q0 = 1 stands in for a Hamiltonian that can't be evaluated there,
such as a Booker root lost to rounding: RK45 then can't take a step and stops.
"""

from scotty.beam_solver import beam_tracing
from scotty.geometry import MagneticField_Cartesian
from scotty.hamiltonian import Hamiltonian

from scotty.logger import config_logger

import numpy as np
import pytest
from numpy.testing import assert_allclose
from unittest.mock import MagicMock


@pytest.fixture(scope="module", autouse=True)
def trace_level(tmp_path_factory):
    """The solvers use `log.trace`, a custom level registered by `config_logger`"""
    config_logger("warning", None, tmp_path_factory.mktemp("log"), "")


Q_NAN = 1.0  # H is NaN for q0 beyond this


class StraightLineHamiltonian(Hamiltonian):
    def __init__(self):
        self.field = MagicMock(spec=MagneticField_Cartesian)
        self.spacings = {name: 1e-4 for name in ("q0", "q1", "q2", "K0", "K1", "K2")}

    def __call__(self, q0, q1, q2, K0, K1, K2):
        return np.where(np.asarray(q0) > Q_NAN, np.nan, K0)


def trace(tau_leave):
    tau_points = np.linspace(0, tau_leave, 21)
    return tau_points, beam_tracing(
        tau_leave=tau_leave,
        tau_points=tau_points,
        q_initial=np.zeros(3),
        K_initial=np.array([1.0, 0.0, 0.0]),
        Psi_3D_initial_labframe=np.diag([1 + 1j, 1 + 1j, 0]).astype(complex),
        hamiltonian=StraightLineHamiltonian(),
        rtol=1e-6,
        atol=1e-9,
    )


def test_beam_solver_succeeds_before_nan():
    """Check of the set-up: short of the NaN region, every point is returned"""
    tau_points, (status, _, _, tau, q, _, Psi) = trace(0.5 * Q_NAN)
    assert status == 0
    assert_allclose(tau, tau_points)
    assert_allclose(q[:, 0], tau_points, atol=1e-6)
    assert_allclose(Psi[-1], Psi[0], atol=1e-6)


def test_beam_solver_failure_raises():
    """The error says how far the beam got, so the user can find where it failed"""
    with pytest.raises(
        RuntimeError, match=r"Beam solver failed .*\(\d+ of 21 points\), last at q = "
    ):
        trace(2 * Q_NAN)
