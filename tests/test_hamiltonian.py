from scotty.hamiltonian import Hamiltonian, hessians
from scotty.fun_general import freq_GHz_to_angular_frequency
from scotty.init_bruv import get_parameters_for_Scotty
from scotty.beam_me_up import create_magnetic_geometry

import numpy as np
from numpy.testing import assert_allclose

import pytest


# Wavenumbers used in the tests
k_q_R = 2.1
k_q_Z = 3.2
k_K_R = 4.3
k_K_zeta = 5.4
k_K_Z = 6.5


class FakeHamiltonian(Hamiltonian):
    def __init__(
        self,
        delta_R: float,
        delta_Z: float,
        delta_K_R: float,
        delta_K_zeta: float,
        delta_K_Z: float,
    ):
        self.spacings = {
            "q_R": delta_R,
            "q_Z": delta_Z,
            "K_R": delta_K_R,
            "K_zeta": delta_K_zeta,
            "K_Z": delta_K_Z,
        }

    def __call__(self, q_R, q_Z, K_R, K_zeta, K_Z):
        return (
            np.exp(k_q_R * q_R)
            * np.exp(k_q_Z * q_Z)
            * np.exp(k_K_R * K_R)
            * np.exp(k_K_zeta * K_zeta)
            * np.exp(k_K_Z * K_Z)
        )


def test_hamiltonian_derivatives():
    H = FakeHamiltonian(1e-3, 1e-3, 1e-4, 1e-4, 1e-4)
    dH = H.derivatives(1.2, 2.3, 3.4, 4.5, 5.6, second_order=True)

    H0 = H(1.2, 2.3, 3.4, 4.5, 5.6)

    dH_dR = dH["dH_dR"] / H0
    dH_dZ = dH["dH_dZ"] / H0
    dH_dKR = dH["dH_dKR"] / H0
    dH_dKzeta = dH["dH_dKzeta"] / H0
    dH_dKZ = dH["dH_dKZ"] / H0
    d2H_dR2 = dH["d2H_dR2"] / H0
    d2H_dZ2 = dH["d2H_dZ2"] / H0
    d2H_dKR2 = dH["d2H_dKR2"] / H0
    d2H_dKzeta2 = dH["d2H_dKzeta2"] / H0
    d2H_dKZ2 = dH["d2H_dKZ2"] / H0
    d2H_dR_dZ = dH["d2H_dR_dZ"] / H0
    d2H_dKR_dR = dH["d2H_dR_dKR"] / H0
    d2H_dKzeta_dR = dH["d2H_dR_dKzeta"] / H0
    d2H_dKZ_dR = dH["d2H_dR_dKZ"] / H0
    d2H_dKR_dZ = dH["d2H_dZ_dKR"] / H0
    d2H_dKzeta_dZ = dH["d2H_dZ_dKzeta"] / H0
    d2H_dKZ_dZ = dH["d2H_dZ_dKZ"] / H0
    d2H_dKR_dKZ = dH["d2H_dKR_dKZ"] / H0
    d2H_dKR_dKzeta = dH["d2H_dKR_dKzeta"] / H0
    d2H_dKzeta_dKZ = dH["d2H_dKzeta_dKZ"] / H0

    assert np.isclose(dH_dR, k_q_R)
    assert np.isclose(dH_dZ, k_q_Z)
    assert np.isclose(dH_dKR, k_K_R)
    assert np.isclose(dH_dKzeta, k_K_zeta)
    assert np.isclose(dH_dKZ, k_K_Z)
    assert np.isclose(d2H_dR2, k_q_R**2)
    assert np.isclose(d2H_dZ2, k_q_Z**2)
    assert np.isclose(d2H_dKR2, k_K_R**2)
    assert np.isclose(d2H_dKzeta2, k_K_zeta**2)
    assert np.isclose(d2H_dKZ2, k_K_Z**2)
    assert np.isclose(d2H_dR_dZ, k_q_R * k_q_Z)
    assert np.isclose(d2H_dKR_dR, k_K_R * k_q_R)
    assert np.isclose(d2H_dKzeta_dR, k_K_zeta * k_q_R)
    assert np.isclose(d2H_dKZ_dR, k_K_Z * k_q_R)
    assert np.isclose(d2H_dKR_dZ, k_K_R * k_q_Z)
    assert np.isclose(d2H_dKzeta_dZ, k_K_zeta * k_q_Z)
    assert np.isclose(d2H_dKZ_dZ, k_K_Z * k_q_Z)
    assert np.isclose(d2H_dKR_dKZ, k_K_R * k_K_Z)
    assert np.isclose(d2H_dKR_dKzeta, k_K_R * k_K_zeta)
    assert np.isclose(d2H_dKzeta_dKZ, k_K_zeta * k_K_Z)


def test_hamiltonian_hessians():
    H = FakeHamiltonian(1e-3, 1e-3, 1e-4, 1e-4, 1e-4)
    dH = H.derivatives(1.2, 2.3, 3.4, 4.5, 5.6, second_order=True)
    H0 = H(1.2, 2.3, 3.4, 4.5, 5.6)

    grad_grad_H, gradK_grad_H, gradK_gradK_H = hessians(dH)

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
    dH_0 = H.derivatives(1.2, 2.3, 3.4, 4.5, 5.6, second_order=True)

    dH_1 = FakeHamiltonian(dx4, dx4, dx4, dx4, dx4).derivatives(
        1.2, 2.3, 3.4, 4.5, 5.6, second_order=True
    )

    H0 = H(1.2, 2.3, 3.4, 4.5, 5.6)

    error_0 = np.abs(dH_0[derivative] / H0 - expected)
    error_1 = np.abs(dH_1[derivative] / H0 - expected)

    err_diff = np.log(error_0 / error_1) / np.log(4)
    assert np.isclose(err_diff, 2, rtol=1e-2)


def test_golden():
    """Check that the actual Hamiltonian is correct

    Very basic golden answer test, could do with analytic answer
    """
    kwargs_dict = get_parameters_for_Scotty("DBS_synthetic")
    kwargs_dict["find_B_method"] = "unit-tests"
    field = create_magnetic_geometry(**kwargs_dict)
    density = kwargs_dict["density_fit_method"]
    angular_frequency = freq_GHz_to_angular_frequency(kwargs_dict["launch_freq_GHz"])
    H = Hamiltonian(
        field,
        angular_frequency,
        kwargs_dict["mode_flag"],
        density,
        1e-3,
        1e-3,
        1e-3,
        1e-3,
        1e-3,
    )
    expected = -0.2632447148279265
    assert np.isclose(H(1.75, 0.1, 1, 1, 1), expected)
