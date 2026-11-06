"""`beam_solver.beam_tracing` raises when `solve_ivp` fails.

A failed integration used to return a truncated output with no error, and an
empty one a `TypeError` from `unpack_beam_parameters`. The right-hand side is
replaced here, so that `solve_ivp` really does fail, partway or at once.
"""

import importlib
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose
import pytest

from scotty.beam_solver import beam_tracing
from scotty.logger import config_logger

beam_solver = importlib.import_module("scotty.beam_solver")

TAU_LEAVE = 10.0
TAU_POINTS = np.linspace(0, TAU_LEAVE, 21)
Q_INITIAL = np.array([1.0, 2.0, 3.0])
K_INITIAL = np.array([4.0, 5.0, 6.0])
PSI_INITIAL = (1.0 + 2.0j) * np.eye(3)


@pytest.fixture(scope="module", autouse=True)
def trace_level(tmp_path_factory):
    """The solvers use `log.trace`, a custom level registered by `config_logger`"""
    config_logger("warning", None, tmp_path_factory.mktemp("log"), "")


def trace(monkeypatch, rhs, tau_points=TAU_POINTS):
    monkeypatch.setattr(beam_solver, "d_beam_parameters_d_tau", rhs)
    return beam_tracing(
        tau_leave=TAU_LEAVE,
        tau_points=tau_points,
        q_initial=Q_INITIAL,
        K_initial=K_INITIAL,
        Psi_3D_initial_labframe=PSI_INITIAL,
        hamiltonian=None,
        rtol=1e-6,
        atol=1e-8,
    )


def test_success_is_unchanged(monkeypatch):
    """A successful integration returns the status and the output as before"""
    status, nfev, _, tau, q, K, Psi = trace(
        monkeypatch, lambda tau, y, hamiltonian: np.ones(18)
    )
    assert status == 0
    assert nfev > 0
    assert_allclose(tau, TAU_POINTS)
    assert_allclose(q[-1], Q_INITIAL + TAU_LEAVE)
    assert_allclose(K[-1], K_INITIAL + TAU_LEAVE)
    assert q.shape == (21, 3)
    assert Psi.shape == (21, 3, 3)


def test_failure_partway_raises(monkeypatch):
    """A singularity at tau = 5.2 stops the solver's steps there. The output would
    have been truncated at tau = 5, the last output point before it"""

    def rhs(tau, y, hamiltonian):
        return np.full(18, 1.0 / (5.2 - tau) ** 2)

    with pytest.raises(RuntimeError) as error:
        trace(monkeypatch, rhs)
    message = str(error.value)
    assert "tau = 5" in message
    assert f"tau_leave = {TAU_LEAVE}" in message
    assert "Required step size is less than spacing between numbers" in message


def test_failure_with_empty_output_raises(monkeypatch):
    """A solver that fails before its first output point, so that `solve_ivp`
    returns empty lists. This used to be
    `TypeError: can't multiply sequence by non-int of type 'complex'`"""

    def rhs(tau, y, hamiltonian):
        return np.full(18, 1.0 / (0.5 - tau) ** 2)

    with pytest.raises(RuntimeError) as error:
        trace(monkeypatch, rhs, tau_points=np.linspace(1.0, TAU_LEAVE, 10))
    message = str(error.value)
    assert "no output points" in message
    assert f"tau_leave = {TAU_LEAVE}" in message
    assert "Required step size is less than spacing between numbers" in message


def test_empty_output_from_a_stubbed_solver(monkeypatch):
    """The solver's own message is passed through"""

    def solve_ivp(**kwargs):
        return SimpleNamespace(
            status=-1,
            nfev=7,
            t=[],
            y=[],
            message="stubbed failure",
        )

    monkeypatch.setattr(beam_solver, "solve_ivp", solve_ivp)
    with pytest.raises(RuntimeError, match="stubbed failure"):
        trace(monkeypatch, lambda tau, y, hamiltonian: np.ones(18))
