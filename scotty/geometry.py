from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.interpolate import RectBivariateSpline

from scotty.typing import ArrayLike


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

    def B_R(self, q_R: ArrayLike, q_Z: ArrayLike) -> ArrayLike:
        q_R, q_Z = np.asfarray(q_R), np.asfarray(q_Z)
        return self.B_p_a * q_Z / (q_R * self.minor_radius_a * self.rho(q_R, q_Z))

    def B_T(self, q_R: ArrayLike, q_Z: ArrayLike) -> ArrayLike:
        q_R, q_Z = np.asfarray(q_R), np.asfarray(q_Z)
        return self.B_T_axis * (self.R_axis / q_R)

    def B_Z(self, q_R: ArrayLike, q_Z: ArrayLike) -> ArrayLike:
        q_R, q_Z = np.asfarray(q_R), np.asfarray(q_Z)
        return (
            -self.B_p_a
            * (q_R - self.R_axis)
            / (q_R * self.minor_radius_a * self.rho(q_R, q_Z))
        )

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


class InterpolatedField(MagneticGeometry):
    """Interpolated numerical equilibrium using bivariate splines"""

    def _make_rect_spline(self, R_grid, Z_grid, array):
        return RectBivariateSpline(
            R_grid,
            Z_grid,
            array,
            bbox=[None, None, None, None],
            kx=self._interp_order,
            ky=self._interp_order,
            s=self._interp_smoothing,
        )

    def __init__(
        self,
        R_grid: npt.ArrayLike,
        Z_grid: npt.ArrayLike,
        B_R: npt.ArrayLike,
        B_T: npt.ArrayLike,
        B_Z: npt.ArrayLike,
        psi: npt.ArrayLike,
        interp_order: int = 5,
        interp_smoothing: int = 0,
    ):
        self._interp_order = interp_order
        self._interp_smoothing = interp_smoothing

        self._interp_B_R = self._make_rect_spline(R_grid, Z_grid, B_R)
        self._interp_B_T = self._make_rect_spline(R_grid, Z_grid, B_T)
        self._interp_B_Z = self._make_rect_spline(R_grid, Z_grid, B_Z)
        self._interp_poloidal_flux = self._make_rect_spline(R_grid, Z_grid, psi)

        self.data_R_coord = R_grid
        self.data_Z_coord = Z_grid
        self.poloidalFlux_grid = psi

    def B_R(self, q_R: ArrayLike, q_Z: ArrayLike) -> ArrayLike:
        return self._interp_B_R(q_R, q_Z, grid=False)

    def B_T(self, q_R: ArrayLike, q_Z: ArrayLike) -> ArrayLike:
        return self._interp_B_T(q_R, q_Z, grid=False)

    def B_Z(self, q_R: ArrayLike, q_Z: ArrayLike) -> ArrayLike:
        return self._interp_B_Z(q_R, q_Z, grid=False)

    def poloidal_flux(self, q_R: ArrayLike, q_Z: ArrayLike) -> ArrayLike:
        return self._interp_poloidal_flux(q_R, q_Z, grid=False)
