from scotty.fun_general import freq_GHz_to_angular_frequency
from scotty.geometry import MagneticField_Cylindrical
from scotty.hamiltonian import Hamiltonian, hessians
from scotty.logger import config_logger
from scotty.profile_fit import QuadraticFit
import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy import constants
from unittest.mock import MagicMock


k_q_R = 2.1
k_q_Z = 3.2
k_K_R = 4.3
k_K_zeta = 5.4
k_K_Z = 6.5

Q_POINT = np.array([1.2, 0.0, 2.3])
K_POINT = np.array([3.4, 4.5, 5.6])


class FakeHamiltonian(Hamiltonian):
    def __init__(self, delta_R, delta_Z, delta_K_R, delta_K_zeta, delta_K_Z):
        self.field = MagicMock(spec=MagneticField_Cylindrical)
        self.spacings = {
            "q0": delta_R, "q1": delta_R, "q2": delta_Z,
            "K0": delta_K_R, "K1": delta_K_zeta, "K2": delta_K_Z,
        }

    def __call__(self, q0, q1, q2, K0, K1, K2):
        return (
            np.exp(k_q_R * q0)
            * np.exp(k_q_Z * q2)
            * np.exp(k_K_R * K0)
            * np.exp(k_K_zeta * K1)
            * np.exp(k_K_Z * K2)
        )


def _H0(H):
    return H(*Q_POINT, *K_POINT)


EXPECTED = {
    "dH_dR": k_q_R,
    "dH_dzeta": 0.0,
    "dH_dZ": k_q_Z,
    "dH_dKR": k_K_R,
    "dH_dKzeta": k_K_zeta,
    "dH_dKZ": k_K_Z,
    "d2H_dR2": k_q_R**2,
    "d2H_dZ2": k_q_Z**2,
    "d2H_dKR2": k_K_R**2,
    "d2H_dKzeta2": k_K_zeta**2,
    "d2H_dKZ2": k_K_Z**2,
    "d2H_dR_dZ": k_q_R * k_q_Z,
    "d2H_dR_dKR": k_K_R * k_q_R,
    "d2H_dR_dKzeta": k_K_zeta * k_q_R,
    "d2H_dR_dKZ": k_K_Z * k_q_R,
    "d2H_dZ_dKR": k_K_R * k_q_Z,
    "d2H_dZ_dKzeta": k_K_zeta * k_q_Z,
    "d2H_dZ_dKZ": k_K_Z * k_q_Z,
    "d2H_dKR_dKZ": k_K_R * k_K_Z,
    "d2H_dKR_dKzeta": k_K_R * k_K_zeta,
    "d2H_dKzeta_dKZ": k_K_zeta * k_K_Z,
}


@pytest.mark.parametrize("key", list(EXPECTED))
def test_hamiltonian_derivatives(key):
    H = FakeHamiltonian(1e-3, 1e-3, 1e-4, 1e-4, 1e-4)
    dH = H.derivatives(Q_POINT, K_POINT, second_order=True)
    assert np.isclose(dH[key] / _H0(H), EXPECTED[key]), key


def test_hamiltonian_derivatives_dzeta_is_zero():
    H = FakeHamiltonian(1e-3, 1e-3, 1e-4, 1e-4, 1e-4)
    dH = H.derivatives(Q_POINT, K_POINT)
    assert_allclose(dH["dH_dzeta"], 0.0)


def test_hamiltonian_hessians():
    H = FakeHamiltonian(1e-3, 1e-3, 1e-4, 1e-4, 1e-4)
    dH = H.derivatives(Q_POINT, K_POINT, second_order=True)
    H0 = _H0(H)
    grad_grad_H, gradK_grad_H, gradK_gradK_H = hessians(dH, cartesian=False)

    grad_grad_H_expected = np.array(
        [[k_q_R**2, 0, k_q_R * k_q_Z], [0, 0, 0], [k_q_R * k_q_Z, 0, k_q_Z**2]]
    )
    gradK_grad_H_expected = np.array(
        [
            [k_K_R * k_q_R, 0, k_K_R * k_q_Z],
            [k_K_zeta * k_q_R, 0, k_K_zeta * k_q_Z],
            [k_K_Z * k_q_R, 0, k_K_Z * k_q_Z],
        ]
    )
    gradK_gradK_H_expected = np.array(
        [
            [k_K_R**2, k_K_R * k_K_zeta, k_K_R * k_K_Z],
            [k_K_zeta * k_K_R, k_K_zeta**2, k_K_zeta * k_K_Z],
            [k_K_Z * k_K_R, k_K_Z * k_K_zeta, k_K_Z**2],
        ]
    )
    assert grad_grad_H.shape == (3, 3)
    assert_allclose(grad_grad_H / H0, grad_grad_H_expected, rtol=1e-5, atol=1e-4)
    assert_allclose(gradK_grad_H / H0, gradK_grad_H_expected, rtol=1e-5, atol=1e-4)
    assert_allclose(gradK_gradK_H / H0, gradK_gradK_H_expected, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize(
    ("derivative", "expected"),
    (
        ("dH_dR", k_q_R),
        ("d2H_dR2", k_q_R**2),
        ("d2H_dR_dZ", k_q_R * k_q_Z),
        ("d2H_dKZ2", k_K_Z**2),
        ("d2H_dKzeta_dKZ", k_K_zeta * k_K_Z),
        ("d2H_dR_dKzeta", k_K_zeta * k_q_R),
    ),
)
def test_hamiltonian_derivatives_scaling(derivative, expected):
    """Check that error scaling of derivatives is second order"""
    dx = 1e-3
    dx4 = dx / 4
    H = FakeHamiltonian(dx, dx, dx, dx, dx)
    dH_0 = H.derivatives(Q_POINT, K_POINT, second_order=True)
    dH_1 = FakeHamiltonian(dx4, dx4, dx4, dx4, dx4).derivatives(
        Q_POINT, K_POINT, second_order=True
    )
    H0 = _H0(H)
    error_0 = np.abs(dH_0[derivative] / H0 - expected)
    error_1 = np.abs(dH_1[derivative] / H0 - expected)
    err_diff = np.log(error_0 / error_1) / np.log(4)
    assert np.isclose(err_diff, 2, rtol=1e-2)


##################################################
# Tests of the real Hamiltonian
##################################################


class ToroidalField(MagneticField_Cylindrical):
    """Purely toroidal field, B_T = B0 R0 / R, with circular flux surfaces"""

    def __init__(self, R0=1.5, a=0.5, B0=1.0):
        self.R0, self.a, self.B0 = R0, a, B0

    def B_R(self, R, _, Z):
        return np.zeros_like(R * Z, dtype=float)

    def B_T(self, R, _, Z):
        return self.B0 * self.R0 / R * np.ones_like(Z, dtype=float)

    def B_Z(self, R, _, Z):
        return np.zeros_like(R * Z, dtype=float)

    def polflux(self, R, _, Z):
        return np.sqrt((R - self.R0) ** 2 + Z**2) / self.a


@pytest.fixture(scope="module", autouse=False)
def trace_level(tmp_path_factory):
    """`Hamiltonian.__call__` uses `log.trace`, a custom log level that is only
    registered by `config_logger`"""
    config_logger("warning", None, tmp_path_factory.mktemp("log"), "")


def _make_hamiltonian(mode_flag):
    return Hamiltonian(
        launch_angular_freq=freq_GHz_to_angular_frequency(55.0),
        mode_flag=mode_flag,
        # cylindrical: (delta_R, delta_Z, delta_K_R, delta_K_zeta, delta_K_Z)
        deltas=np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-3]),
        field=ToroidalField(),
        density_fit=QuadraticFit(1.0, 4.0),
    )


def _appleton_hartree_H(mode_flag, R, Z, K_R, K_zeta, K_Z):
    """Independent cold-plasma Appleton-Hartree reference for `ToroidalField`
    and `QuadraticFit(1.0, 4.0)` at 55 GHz"""
    w = freq_GHz_to_angular_frequency(55.0)
    K_vec = np.array([K_R, K_zeta / R, K_Z])
    K2 = K_vec @ K_vec
    cos2 = K_vec[1] ** 2 / K2  # B is purely toroidal, so b.K = K_zeta / R
    sin2 = 1 - cos2

    psi = np.sqrt((R - 1.5) ** 2 + Z**2) / 0.5
    ne = 4.0 * (1 - psi**2)  # in 1e19 m^-3
    X = constants.e**2 * ne * 1e19 / (constants.epsilon_0 * constants.m_e) / w**2
    Y = constants.e * (1.5 / R) / constants.m_e / w

    root = np.sqrt(Y**4 * sin2**2 + 4 * (1 - X) ** 2 * Y**2 * cos2)
    den = 2 * (1 - X) - Y**2 * sin2 + mode_flag * root
    n2 = 1 - 2 * X * (1 - X) / den
    k0 = w / constants.c
    return K2 / k0**2 - n2


@pytest.mark.parametrize("mode_flag", (1, -1))
@pytest.mark.parametrize(
    "point",
    (
        (1.75, 0.0, 0.1, 1.0, 1.0, 1.0),
        (1.6, 0.0, -0.2, 500.0, 300.0, 250.0),
    ),
)
def test_hamiltonian_matches_appleton_hartree(trace_level, mode_flag, point):
    """Check the real Hamiltonian against an independent analytic dispersion relation"""
    H = _make_hamiltonian(mode_flag)
    R, zeta, Z, K_R, K_zeta, K_Z = point
    assert np.isclose(H(*point), _appleton_hartree_H(mode_flag, R, Z, K_R, K_zeta, K_Z))


def test_golden(trace_level):
    """Golden answer: O-mode on `ToroidalField` at a single point

    Value generated with the current implementation and cross-checked
    against the analytic Appleton-Hartree result above
    """
    H = _make_hamiltonian(1)
    assert np.isclose(H(1.75, 0.0, 0.1, 1, 1, 1), -0.27056056863516875)


def test_two_argument_call(trace_level):
    """H(q, K) should give the same answer as H(q0, q1, q2, K0, K1, K2)"""
    H = _make_hamiltonian(1)
    q = np.array([1.75, 0.0, 0.1])
    K = np.array([1.0, 1.0, 1.0])
    assert np.isclose(H(q, K), H(*q, *K))


def test_wrong_number_of_arguments_raises(trace_level):
    H = _make_hamiltonian(1)
    with pytest.raises(RuntimeError):
        H(1.75, 0.0, 0.1, 1.0)