from __future__ import annotations

from abc import ABC
from typing import Callable, Optional

import numpy as np
import numpy.typing as npt
from scipy.interpolate import RectBivariateSpline, UnivariateSpline

from scotty.fun_CFD import find_dpolflux_dR, find_dpolflux_dZ
from scotty.typing import ArrayLike, FloatArray


class MagneticField(ABC):
    """Abstract base class for magnetic field geometries

    Attributes
    ----------
    R_coord:
    Z_coord:
        Arrays with the (original) sample locations for the poloidal
        flux and magnetic field components
    """

    R_coord: FloatArray
    Z_coord: FloatArray

    def B_R(self, q_R: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        raise NotImplementedError

    def B_T(self, q_R: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        raise NotImplementedError

    def B_Z(self, q_R: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        raise NotImplementedError

    def poloidal_flux(self, q_R: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        raise NotImplementedError


class CircularCrossSectionField(MagneticField):
    """Simple circular cross-section magnetic geometry

    Parameters
    ----------
    B_T_axis:
        Toroidal magnetic field at the magnetic axis (Tesla)
    R_axis:
        Major radius of the magnetic axis (metres)
    minor_radius_a:
        Minor radius of the last closed flux surface (metres)
    B_p_a:
        Poloidal magnetic field at ``minor_radius_a`` (Tesla)
    R_points:
    Z_points:
        Number of points for sample ``(R, Z)`` grid
    grid_buffer_factor:
        Multiplicative factor to increase size of sample grid by
    """

    def __init__(
        self,
        B_T_axis: float,
        R_axis: float,
        minor_radius_a: float,
        B_p_a: float,
        R_points: int = 101,
        Z_points: int = 101,
        grid_buffer_factor: float = 1.0,
    ):
        self.B_T_axis = B_T_axis
        self.R_axis = R_axis
        self.minor_radius_a = minor_radius_a
        self.B_p_a = B_p_a

        grid_width = grid_buffer_factor * minor_radius_a
        self.R_coord = np.linspace(R_axis - grid_width, R_axis + grid_width, R_points)
        self.Z_coord = np.linspace(-grid_width, grid_width, Z_points)
        self.poloidalFlux_grid = self.poloidal_flux(
            *np.meshgrid(self.R_coord, self.Z_coord, indexing="ij")
        )

    def rho(self, q_R: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        q_R, q_Z = np.asfarray(q_R), np.asfarray(q_Z)
        return np.sqrt((q_R - self.R_axis) ** 2 + q_Z**2)

    def B_R(self, q_R: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        q_R, q_Z = np.asfarray(q_R), np.asfarray(q_Z)
        return self.B_p_a * q_Z / (q_R * self.minor_radius_a * self.rho(q_R, q_Z))

    def B_T(self, q_R: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        q_R, q_Z = np.asfarray(q_R), np.asfarray(q_Z)
        return self.B_T_axis * (self.R_axis / q_R)

    def B_Z(self, q_R: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        q_R, q_Z = np.asfarray(q_R), np.asfarray(q_Z)
        return (
            -self.B_p_a
            * (q_R - self.R_axis)
            / (q_R * self.minor_radius_a * self.rho(q_R, q_Z))
        )

    def poloidal_flux(self, q_R: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        q_R, q_Z = np.asfarray(q_R), np.asfarray(q_Z)
        return self.rho(q_R, q_Z) / self.minor_radius_a


class ConstantCurrentDensityField(MagneticField):
    """Circular cross-section magnetic geometry with constant current density

    TODO
    ----
    Poloidal flux needs to be made consistent with  B_{R,Z}

    Parameters
    ----------
    B_T_axis:
        Toroidal magnetic field at the magnetic axis (Tesla)
    R_axis:
        Major radius of the magnetic axis (metres)
    minor_radius_a:
        Minor radius of the last closed flux surface (metres)
    B_p_a:
        Poloidal magnetic field at ``minor_radius_a`` (Tesla)
    R_points:
    Z_points:
        Number of points for sample ``(R, Z)`` grid
    grid_buffer_factor:
        Multiplicative factor to increase size of sample grid by
    """

    def __init__(
        self,
        B_T_axis: float,
        R_axis: float,
        minor_radius_a: float,
        B_p_a: float,
        R_points: int = 101,
        Z_points: int = 101,
        grid_buffer_factor: float = 1,
    ):
        self.B_T_axis = B_T_axis
        self.R_axis = R_axis
        self.minor_radius_a = minor_radius_a
        self.B_p_a = B_p_a

        grid_width = grid_buffer_factor * minor_radius_a
        self.R_coord = np.linspace(R_axis - grid_width, R_axis + grid_width, R_points)
        self.Z_coord = np.linspace(-grid_width, grid_width, Z_points)
        self.poloidalFlux_grid = self.poloidal_flux(
            *np.meshgrid(self.R_coord, self.Z_coord, indexing="ij")
        )

    def rho(self, q_R: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        q_R, q_Z = np.asfarray(q_R), np.asfarray(q_Z)
        return np.sqrt((q_R - self.R_axis) ** 2 + q_Z**2)

    def B_R(self, q_R: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        q_R, q_Z = np.asfarray(q_R), np.asfarray(q_Z)
        return self.B_p_a * q_Z / self.rho(q_R, q_Z)

    def B_T(self, q_R: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        q_R, q_Z = np.asfarray(q_R), np.asfarray(q_Z)
        return self.B_T_axis * (self.R_axis / q_R)

    def B_Z(self, q_R: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        q_R, q_Z = np.asfarray(q_R), np.asfarray(q_Z)
        return -self.B_p_a * (q_R - self.R_axis) / self.rho(q_R, q_Z)

    def poloidal_flux(self, q_R: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        q_R, q_Z = np.asfarray(q_R), np.asfarray(q_Z)
        return self.rho(q_R, q_Z) / self.minor_radius_a


class CurvySlabField(MagneticField):
    """Analytical curvy slab geometry"""

    def __init__(self, B_T_axis: float, R_axis: float):
        self.B_T_axis = B_T_axis
        self.R_axis = R_axis

    def B_R(self, q_R: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        q_R, q_Z = np.asfarray(q_R), np.asfarray(q_Z)
        return np.zeros_like(q_R)

    def B_T(self, q_R: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        q_R, q_Z = np.asfarray(q_R), np.asfarray(q_Z)
        return self.B_T_axis * self.R_axis / q_R

    def B_Z(self, q_R: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        q_R, q_Z = np.asfarray(q_R), np.asfarray(q_Z)
        return np.zeros_like(q_R)


def _make_rect_spline(
    R_grid, Z_grid, array, interp_order: int, interp_smoothing: int
) -> Callable[[ArrayLike, ArrayLike], FloatArray]:
    spline = RectBivariateSpline(
        R_grid,
        Z_grid,
        array,
        bbox=[None, None, None, None],
        kx=interp_order,
        ky=interp_order,
        s=interp_smoothing,
    )
    return lambda R, Z: spline(R, Z, grid=False)


class InterpolatedField(MagneticField):
    """Interpolated numerical equilibrium using bivariate splines

    Parameters
    ----------
    R_grid:
        1D array of points in ``R`` (metres)
    Z_grid:
        1D array of points in ``Z`` (metres)
    B_R:
        2D ``(R, Z)`` grid of radial magnetic field values (Tesla)
    B_T:
        2D ``(R, Z)`` grid of toroidal magnetic field values (Tesla)
    B_Z:
        2D ``(R, Z)`` grid of vertical magnetic field values (Tesla)
    psi:
        2D ``(R, Z)`` grid of poloidal flux values (Weber/radian)
    interp_order:
        Order of interpolating splines
    interp_smoothing:
        Smoothing factor for interpolating splines
    """

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
        self._interp_B_R = _make_rect_spline(
            R_grid, Z_grid, B_R, interp_order, interp_smoothing
        )
        self._interp_B_T = _make_rect_spline(
            R_grid, Z_grid, B_T, interp_order, interp_smoothing
        )
        self._interp_B_Z = _make_rect_spline(
            R_grid, Z_grid, B_Z, interp_order, interp_smoothing
        )
        self._interp_poloidal_flux = _make_rect_spline(
            R_grid, Z_grid, psi, interp_order, interp_smoothing
        )

        self.R_coord = R_grid
        self.Z_coord = Z_grid
        self.poloidalFlux_grid = psi

    def B_R(self, q_R: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        return self._interp_B_R(q_R, q_Z)

    def B_T(self, q_R: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        return self._interp_B_T(q_R, q_Z)

    def B_Z(self, q_R: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        return self._interp_B_Z(q_R, q_Z)

    def poloidal_flux(self, q_R: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        return self._interp_poloidal_flux(q_R, q_Z)


class EFITField(MagneticField):
    def __init__(
        self,
        R_grid: npt.ArrayLike,
        Z_grid: npt.ArrayLike,
        rBphi: npt.NDArray[np.float64],
        psi_norm_2D: npt.ArrayLike,
        psi_unnorm_axis: float,
        psi_unnorm_boundary: float,
        psi_norm_1D: Optional[npt.ArrayLike] = None,
        delta_R: float = 0.0001,
        delta_Z: float = 0.0001,
        interp_order: int = 5,
        interp_smoothing: int = 0,
    ):
        self.R_coord = R_grid
        self.Z_coord = Z_grid
        self.poloidalFlux_grid = psi_norm_2D

        self.delta_R = delta_R
        self.delta_Z = delta_Z

        self._interp_poloidal_flux = _make_rect_spline(
            R_grid, Z_grid, psi_norm_2D, interp_order, interp_smoothing
        )

        self.poloidal_flux_gradient = psi_unnorm_boundary - psi_unnorm_axis
        if psi_norm_1D is None:
            psi_norm_1D = np.linspace(0, 1.0, len(rBphi))

        self._interp_rBphi = UnivariateSpline(
            psi_norm_1D,
            rBphi,
            w=None,
            bbox=[None, None],
            k=interp_order,
            s=interp_smoothing,
            ext=0,
            check_finite=False,
        )

    def B_R(self, q_R: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        dpolflux_dZ = find_dpolflux_dZ(
            q_R, q_Z, self.delta_Z, self._interp_poloidal_flux
        )
        return -dpolflux_dZ * self.poloidal_flux_gradient / q_R

    def B_T(self, q_R: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        polflux = self._interp_poloidal_flux(q_R, q_Z)
        return self._interp_rBphi(polflux) / q_R

    def B_Z(self, q_R: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        dpolflux_dR = find_dpolflux_dR(
            q_R, q_Z, self.delta_R, self._interp_poloidal_flux
        )
        return dpolflux_dR * self.poloidal_flux_gradient / q_R

    def poloidal_flux(self, q_R: ArrayLike, q_Z: ArrayLike) -> FloatArray:
        return self._interp_poloidal_flux(q_R, q_Z)
