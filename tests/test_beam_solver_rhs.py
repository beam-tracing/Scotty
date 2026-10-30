"""The ray part of the beam-solver right-hand side, in both geometries.

With H = exp(k . (q, K)), every derivative is H times the matching k, so
dq/dtau = dH/dK and dK/dtau = -dH/dq are known exactly. Mixing up a component,
or reading the derivatives from the dictionary in the wrong order, shows up
as the wrong k.
"""

from scotty.beam_solver import d_beam_parameters_d_tau, unpack_beam_parameters
from scotty.geometry import MagneticField_Cartesian, MagneticField_Cylindrical
from scotty.hamiltonian import FIRST_ORDER_KEYS, Hamiltonian
from scotty.ray_solver import d_ray_parameters_d_tau

from scotty.logger import config_logger

import numpy as np
import pytest
from numpy.testing import assert_allclose
from unittest.mock import MagicMock


@pytest.fixture(scope="module", autouse=True)
def trace_level(tmp_path_factory):
    """The solvers use `log.trace`, a custom level registered by `config_logger`"""
    config_logger("warning", None, tmp_path_factory.mktemp("log"), "")

K_Q = np.array([1.2, 0.0, 2.1])  # no zeta dependence, as in an axisymmetric field
K_K = np.array([4.3, 5.4, 6.5])
Q_POINT = np.array([1.2, 0.3, 2.3])
K_POINT = np.array([3.4, 4.5, 5.6])


class ExponentialHamiltonian(Hamiltonian):
    def __init__(self, geometry):
        spec = MagneticField_Cartesian if geometry == "cartesian" else MagneticField_Cylindrical
        self.field = MagicMock(spec=spec)
        self.spacings = {name: 1e-4 for name in ("q0", "q1", "q2", "K0", "K1", "K2")}

    def __call__(self, q0, q1, q2, K0, K1, K2):
        return np.exp(K_Q[0] * q0 + K_Q[2] * q2 + K_K[0] * K0 + K_K[1] * K1 + K_K[2] * K2)


@pytest.mark.parametrize("geometry", ["cylindrical", "cartesian"])
def test_first_order_keys_match_derivatives(geometry):
    """The named first-order keys exist, in coordinate order, in the dictionary"""
    dH = ExponentialHamiltonian(geometry).derivatives(Q_POINT, K_POINT)
    q_keys, K_keys = FIRST_ORDER_KEYS[geometry]
    assert tuple(dH)[:6] == q_keys + K_keys


@pytest.mark.parametrize("geometry", ["cylindrical", "cartesian"])
def test_ray_equations(geometry):
    """dq/dtau = dH/dK and dK/dtau = -dH/dq, component by component"""
    H = ExponentialHamiltonian(geometry)
    Psi = np.zeros((3, 3), dtype=complex)
    state = np.concatenate([Q_POINT, K_POINT, np.zeros(12)])

    dq_dtau, dK_dtau, _ = unpack_beam_parameters(d_beam_parameters_d_tau(0.0, state, H))
    H0 = H(*Q_POINT, *K_POINT)

    assert_allclose(dq_dtau / H0, K_K, rtol=1e-6)
    assert_allclose(dK_dtau / H0, -K_Q, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("geometry", ["cylindrical", "cartesian"])
def test_ray_equations_ray_solver(geometry):
    """The ray solver's right-hand side is the same map, on a 6-vector (q, K)"""
    H = ExponentialHamiltonian(geometry)
    rhs = d_ray_parameters_d_tau(0.0, np.concatenate([Q_POINT, K_POINT]), H)
    H0 = H(*Q_POINT, *K_POINT)

    assert_allclose(rhs[:3] / H0, K_K, rtol=1e-6)
    assert_allclose(rhs[3:] / H0, -K_Q, rtol=1e-6, atol=1e-9)
