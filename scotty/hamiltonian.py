from scotty.geometry import MagneticField
from scotty.density_fit import DensityFitLike
from scotty.fun_general import (
    angular_frequency_to_wavenumber,
    find_normalised_gyro_freq,
    find_normalised_plasma_freq,
    contract_special,
)
from scotty.typing import ArrayLike, FloatArray


from dataclasses import dataclass, asdict
import numpy as np
from typing import Dict


class PolarisationTensor:
    def __init__(self, electron_density, angular_frequency, B_total):
        _plasma_freq_2 = (
            find_normalised_plasma_freq(electron_density, angular_frequency) ** 2
        )
        _gyro_freq = find_normalised_gyro_freq(B_total, angular_frequency)
        _gyro_freq_2 = _gyro_freq**2

        self._epsilon_bb = 1 - _plasma_freq_2
        self._epsilon_11 = 1 - _plasma_freq_2 / (1 - _gyro_freq_2)
        self._epsilon_12 = _plasma_freq_2 * _gyro_freq / (1 - _gyro_freq_2)

    @property
    def parallel(self):
        return self._epsilon_bb

    @property
    def perpendicular(self):
        return self._epsilon_11

    @property
    def g(self):
        return self._epsilon_12


FFD1_stencil = {(0,): -3 / 2, (1,): 2, (2,): -0.5}
FFD2_stencil = {(0,): 2, (1,): -5, (2,): 4, (3,): -1}
CFD1_stencil = {(-1,): -0.5, (1,): 0.5}
CFD2_stencil = {(-1,): 1, (0,): -2, (1,): 1}
FFD_FFD_stencil = {
    (0, 0): 2.25,
    (0, 1): -3,
    (0, 2): 0.75,
    (1, 0): -3,
    (1, 1): 4,
    (1, 2): -1,
    (2, 0): 0.75,
    (2, 1): -1,
    (2, 2): 0.25,
}
FFD_CFD_stencil = {
    (0, -1): 0.25,
    (0, 0): 1,
    (0, 1): -1.25,
    (1, 0): -2,
    (1, 1): 2,
    (2, -1): -0.25,
    (2, 0): 1,
    (2, 1): -0.75,
}
CFD_CFD_stencil = {(1, 1): 0.25, (1, -1): -0.25, (-1, 1): -0.25, (-1, -1): 0.25}


@dataclass(frozen=True)
class CoordOffset:
    q_R: int = 0
    q_Z: int = 0
    K_R: int = 0
    K_zeta: int = 0
    K_Z: int = 0


class Hamiltonian:
    def __init__(
        self,
        field: MagneticField,
        launch_angular_frequency: float,
        mode_flag: int,
        density_fit: DensityFitLike,
        delta_R: float,
        delta_Z: float,
        delta_K_R: float,
        delta_K_zeta: float,
        delta_K_Z: float,
    ):
        self.field = field
        self.angular_frequency = launch_angular_frequency
        self.wavenumber_K0 = angular_frequency_to_wavenumber(launch_angular_frequency)
        self.mode_flag = mode_flag
        self.density = density_fit
        self.spacings = {
            "q_R": delta_R,
            "q_Z": delta_Z,
            "K_R": delta_K_R,
            "K_zeta": delta_K_zeta,
            "K_Z": delta_K_Z,
        }

    def __call__(self, q_R, q_Z, K_R, K_zeta, K_Z):
        K_magnitude = np.sqrt(K_R**2 + (K_zeta / q_R) ** 2 + K_Z**2)
        poloidal_flux = self.field.poloidal_flux(q_R, q_Z)
        electron_density = self.density(poloidal_flux)
        B_R = np.squeeze(self.field.B_R(q_R, q_Z))
        B_T = np.squeeze(self.field.B_T(q_R, q_Z))
        B_Z = np.squeeze(self.field.B_Z(q_R, q_Z))

        B_total = np.sqrt(B_R**2 + B_T**2 + B_Z**2)
        b_hat = np.array([B_R, B_T, B_Z]) / B_total
        K_hat = np.array([K_R, K_zeta / q_R, K_Z]) / K_magnitude

        # square of the mismatch angle
        if np.size(q_R) == 1:
            sin_theta_m_sq = np.dot(b_hat, K_hat) ** 2
        else:  # Vectorised version of find_H
            b_hat = b_hat.T
            K_hat = K_hat.T
            sin_theta_m_sq = contract_special(b_hat, K_hat) ** 2

        epsilon = PolarisationTensor(electron_density, self.angular_frequency, B_total)

        Booker_alpha = (epsilon.parallel * sin_theta_m_sq) + epsilon.perpendicular * (
            1 - sin_theta_m_sq
        )
        Booker_beta = (
            -epsilon.perpendicular * epsilon.parallel * (1 + sin_theta_m_sq)
        ) - (epsilon.perpendicular**2 - epsilon.g**2) * (1 - sin_theta_m_sq)
        Booker_gamma = epsilon.parallel * (epsilon.perpendicular**2 - epsilon.g**2)

        H_discriminant = np.maximum(
            np.zeros_like(Booker_beta),
            (Booker_beta**2 - 4 * Booker_alpha * Booker_gamma),
        )

        return (K_magnitude / self.wavenumber_K0) ** 2 + (
            Booker_beta - self.mode_flag * np.sqrt(H_discriminant)
        ) / (2 * Booker_alpha)

    def derivatives(
        self,
        q_R: ArrayLike,
        q_Z: ArrayLike,
        K_R: ArrayLike,
        K_zeta: ArrayLike,
        K_Z: ArrayLike,
        second_order: bool = False,
    ):
        cache: Dict[CoordOffset, ArrayLike] = {}

        starts = {"q_R": q_R, "q_Z": q_Z, "K_R": K_R, "K_zeta": K_zeta, "K_Z": K_Z}

        def apply_stencil(dims, stencil, order=1):
            result = 0.0

            dim_spacings = [self.spacings[dim] for dim in dims] * order

            for stencil_offsets, weight in stencil.items():
                coord_offsets = CoordOffset(**dict(zip(dims, stencil_offsets)))
                try:
                    H_s0_s1 = cache[coord_offsets]
                except KeyError:
                    offsets = asdict(coord_offsets)
                    coords = {
                        dim: start + offsets[dim] * self.spacings[dim]
                        for dim, start in starts.items()
                    }
                    H_s0_s1 = self(**coords)
                    cache[coord_offsets] = H_s0_s1

                result += weight * H_s0_s1
            return result / np.prod(dim_spacings)

        derivatives = {
            "dH_dR": apply_stencil(("q_R",), FFD1_stencil),
            "dH_dZ": apply_stencil(("q_Z",), FFD1_stencil),
            "dH_dKR": apply_stencil(("K_R",), CFD1_stencil),
            "dH_dKzeta": apply_stencil(("K_zeta",), CFD1_stencil),
            "dH_dKZ": apply_stencil(("K_Z",), CFD1_stencil),
        }

        if second_order:
            second_derivatives = {
                "d2H_dR2": apply_stencil(("q_R",), FFD2_stencil, order=2),
                "d2H_dZ2": apply_stencil(("q_Z",), FFD2_stencil, order=2),
                "d2H_dKR2": apply_stencil(("K_R",), CFD2_stencil, order=2),
                "d2H_dKzeta2": apply_stencil(("K_zeta",), CFD2_stencil, order=2),
                "d2H_dKZ2": apply_stencil(("K_Z",), CFD2_stencil, order=2),
                "d2H_dR_dZ": apply_stencil(("q_R", "q_Z"), FFD_FFD_stencil),
                "d2H_dR_dKR": apply_stencil(("q_R", "K_R"), FFD_CFD_stencil),
                "d2H_dR_dKzeta": apply_stencil(("q_R", "K_zeta"), FFD_CFD_stencil),
                "d2H_dR_dKZ": apply_stencil(("q_R", "K_Z"), FFD_CFD_stencil),
                "d2H_dZ_dKR": apply_stencil(("q_Z", "K_R"), FFD_CFD_stencil),
                "d2H_dZ_dKzeta": apply_stencil(("q_Z", "K_zeta"), FFD_CFD_stencil),
                "d2H_dZ_dKZ": apply_stencil(("q_Z", "K_Z"), FFD_CFD_stencil),
                "d2H_dKR_dKZ": apply_stencil(("K_R", "K_Z"), CFD_CFD_stencil),
                "d2H_dKR_dKzeta": apply_stencil(("K_R", "K_zeta"), CFD_CFD_stencil),
                "d2H_dKzeta_dKZ": apply_stencil(("K_zeta", "K_Z"), CFD_CFD_stencil),
            }
            derivatives.update(second_derivatives)
        return derivatives


def laplacians(dH: dict):
    r"""Compute the Laplacians of the Hamiltonian:

    .. math::
        \grad \grad H
        \grad_K \grad H
        \grad_K \grad_K H

    given a ``dict`` containing the second derivatives of the Hamiltonian
    """

    d2H_dR2 = dH["d2H_dR2"]
    d2H_dZ2 = dH["d2H_dZ2"]
    d2H_dKR2 = dH["d2H_dKR2"]
    d2H_dKzeta2 = dH["d2H_dKzeta2"]
    d2H_dKZ2 = dH["d2H_dKZ2"]
    d2H_dR_dZ = dH["d2H_dR_dZ"]
    d2H_dKR_dR = dH["d2H_dR_dKR"]
    d2H_dKzeta_dR = dH["d2H_dR_dKzeta"]
    d2H_dKZ_dR = dH["d2H_dR_dKZ"]
    d2H_dKR_dZ = dH["d2H_dZ_dKR"]
    d2H_dKzeta_dZ = dH["d2H_dZ_dKzeta"]
    d2H_dKZ_dZ = dH["d2H_dZ_dKZ"]
    d2H_dKR_dKZ = dH["d2H_dKR_dKZ"]
    d2H_dKR_dKzeta = dH["d2H_dKR_dKzeta"]
    d2H_dKzeta_dKZ = dH["d2H_dKzeta_dKZ"]

    zeros = np.zeros_like(d2H_dR2)

    def reshape(array: FloatArray):
        """Such that shape is [points,3,3] instead of [3,3,points]"""
        if array.ndim == 2:
            return array
        return np.moveaxis(np.squeeze(array), 2, 0)

    grad_grad_H = reshape(
        np.array(
            [
                [d2H_dR2, zeros, d2H_dR_dZ],
                [zeros, zeros, zeros],
                [d2H_dR_dZ, zeros, d2H_dZ2],
            ]
        )
    )
    gradK_grad_H = reshape(
        np.array(
            [
                [d2H_dKR_dR, zeros, d2H_dKR_dZ],
                [d2H_dKzeta_dR, zeros, d2H_dKzeta_dZ],
                [d2H_dKZ_dR, zeros, d2H_dKZ_dZ],
            ]
        )
    )
    gradK_gradK_H = reshape(
        np.array(
            [
                [d2H_dKR2, d2H_dKR_dKzeta, d2H_dKR_dKZ],
                [d2H_dKR_dKzeta, d2H_dKzeta2, d2H_dKzeta_dKZ],
                [d2H_dKR_dKZ, d2H_dKzeta_dKZ, d2H_dKZ2],
            ]
        )
    )
    return grad_grad_H, gradK_grad_H, gradK_gradK_H
