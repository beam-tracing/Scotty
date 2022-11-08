from __future__ import annotations
import functools

import numpy as np

from numpy.typing import ArrayLike


class MagneticGeometry:
    def B_R(self, q_R: ArrayLike, q_Z: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    def B_T(self, q_R: ArrayLike, q_Z: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    def B_Z(self, q_R: ArrayLike, q_Z: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    def poloidal_flux(self, q_R: ArrayLike, q_Z: ArrayLike) -> ArrayLike:
        raise NotImplementedError


class CircularCrossSection(MagneticGeometry):
    def __init__(
        self, B_T_axis: float, R_axis: float, minor_radius_a: float, B_p_a: float
    ):
        self.B_T_axis = B_T_axis
        self.R_axis = R_axis
        self.minor_radius_a = minor_radius_a
        self.B_p_a = B_p_a

    def rho(self, q_R: ArrayLike, q_Z: ArrayLike) -> ArrayLike:
        q_R, q_Z = np.asfarray(q_R), np.asfarray(q_Z)
        return np.sqrt((q_R - self.R_axis) ** 2 + q_Z**2)

    def B_p(self, q_R: ArrayLike, q_Z: ArrayLike) -> ArrayLike:
        q_R, q_Z = np.asfarray(q_R), np.asfarray(q_Z)
        return -self.B_p_a * (self.rho(q_R, q_Z) / self.minor_radius_a)

    def B_R(self, q_R: ArrayLike, q_Z: ArrayLike) -> ArrayLike:
        q_R, q_Z = np.asfarray(q_R), np.asfarray(q_Z)
        return self.B_p(q_R, q_Z) * (q_Z / self.rho(q_R, q_Z))

    def B_T(self, q_R: ArrayLike, q_Z: ArrayLike) -> ArrayLike:
        q_R, q_Z = np.asfarray(q_R), np.asfarray(q_Z)
        return self.B_T_axis * (self.R_axis / q_R)

    def B_Z(self, q_R: ArrayLike, q_Z: ArrayLike) -> ArrayLike:
        q_R, q_Z = np.asfarray(q_R), np.asfarray(q_Z)
        return -self.B_p(q_R, q_Z) * (q_R - self.R_axis) / self.rho(q_R, q_Z)

    def poloidal_flux(self, q_R: ArrayLike, q_Z: ArrayLike) -> ArrayLike:
        q_R, q_Z = np.asfarray(q_R), np.asfarray(q_Z)
        return self.rho(q_R, q_Z) / self.minor_radius_a


class CurvySlab(MagneticGeometry):
    """Analytical curvy slab geometry"""

    def __init__(self, B_T_axis: float, R_axis: float):
        self.B_T_axis = B_T_axis
        self.R_axis = R_axis

    def B_R(self, q_R: ArrayLike, q_Z: ArrayLike) -> ArrayLike:
        q_R, q_Z = np.asfarray(q_R), np.asfarray(q_Z)
        return np.zeros_like(q_R)

    def B_T(self, q_R: ArrayLike, q_Z: ArrayLike) -> ArrayLike:
        q_R, q_Z = np.asfarray(q_R), np.asfarray(q_Z)
        return self.B_T_axis * self.R_axis / q_R

    def B_Z(self, q_R: ArrayLike, q_Z: ArrayLike) -> ArrayLike:
        q_R, q_Z = np.asfarray(q_R), np.asfarray(q_Z)
        return np.zeros_like(q_R)
