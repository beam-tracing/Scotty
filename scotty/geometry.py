from __future__ import annotations

from abc import ABC
import pathlib
from typing import Callable, Optional

from netCDF4 import Dataset
import numpy as np
import numpy.typing as npt
from scipy.interpolate import RectBivariateSpline, UnivariateSpline

from scotty.fun_CFD import find_dpolflux_dR, find_dpolflux_dZ
from scotty.fun_general import find_nearest
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
    poloidalFlux_grid: FloatArray

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
    return lambda q_R, q_Z: spline(q_R, q_Z, grid=False)


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
        R_grid: FloatArray,
        Z_grid: FloatArray,
        B_R: FloatArray,
        B_T: FloatArray,
        B_Z: FloatArray,
        psi: FloatArray,
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
        R_grid: FloatArray,
        Z_grid: FloatArray,
        rBphi: FloatArray,
        psi_norm_2D: FloatArray,
        psi_unnorm_axis: float,
        psi_unnorm_boundary: float,
        psi_norm_1D: Optional[FloatArray] = None,
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

    @classmethod
    def from_EFITpp(
        cls,
        filename: pathlib.Path,
        equil_time: float,
        delta_R: float,
        delta_Z: float,
        interp_order: int,
        interp_smoothing: int,
    ):
        with Dataset(filename) as dataset:
            efitpp_times = dataset.variables["time"][:]
            time_idx = find_nearest(efitpp_times, equil_time)
            print("EFIT++ time", efitpp_times[time_idx])

            output_group = dataset.groups["output"]
            profiles2D = output_group.groups["profiles2D"]
            # unnormalised, as a function of R and Z
            unnorm_psi_2D = profiles2D.variables["poloidalFlux"][time_idx][:][:]
            data_R_coord = profiles2D.variables["r"][time_idx][:]
            data_Z_coord = profiles2D.variables["z"][time_idx][:]

            fluxFunctionProfiles = output_group.groups["fluxFunctionProfiles"]
            norm_psi_1D = fluxFunctionProfiles.variables["normalizedPoloidalFlux"][:]
            # poloidalFlux as a function of normalised poloidal flux
            unorm_psi_1D = fluxFunctionProfiles.variables["poloidalFlux"][time_idx][:]
            rBphi = fluxFunctionProfiles.variables["rBphi"][time_idx][:]

        # linear fit
        polflux_const_m, polflux_const_c = np.polyfit(unorm_psi_1D, norm_psi_1D, 1)
        norm_psi_2D = unnorm_psi_2D * polflux_const_m + polflux_const_c

        return EFITField(
            R_grid=data_R_coord,
            Z_grid=data_Z_coord,
            rBphi=rBphi,
            psi_norm_2D=norm_psi_2D,
            psi_norm_1D=norm_psi_1D,
            psi_unnorm_axis=0.0,
            psi_unnorm_boundary=polflux_const_m,
            delta_R=delta_R,
            delta_Z=delta_Z,
            interp_order=interp_order,
            interp_smoothing=interp_smoothing,
        )

    @classmethod
    def from_MAST_saved(
        cls,
        filename: pathlib.Path,
        equil_time: float,
        delta_R: float,
        delta_Z: float,
        interp_order: int,
        interp_smoothing: int,
    ):
        with np.load(filename) as loadfile:
            # On time base C
            rBphi_all_times = loadfile["rBphi"]
            t_base_B = loadfile["t_base_B"]
            t_base_C = loadfile["t_base_C"]
            data_R_coord = loadfile["R_EFIT"]
            data_Z_coord = loadfile["Z_EFIT"]
            # Time base C
            polflux_axis_all_times = loadfile["poloidal_flux_unnormalised_axis"]
            # Time base C
            psi_boundary_all_times = loadfile["poloidal_flux_unnormalised_boundary"]
            # On time base B
            unnorm_psi_all_times = loadfile["poloidal_flux_unnormalised"]

        t_base_B_idx = find_nearest(t_base_B, equil_time)
        # Get the same time slice
        t_base_C_idx = find_nearest(t_base_C, t_base_B[t_base_B_idx])
        print("EFIT time", t_base_B[t_base_B_idx])

        rBphi = rBphi_all_times[t_base_C_idx, :]
        psi_axis = polflux_axis_all_times[t_base_C_idx]
        psi_boundary = psi_boundary_all_times[t_base_C_idx]
        unnorm_psi_2D = unnorm_psi_all_times[t_base_B_idx, :, :].T

        # Taken from an old file of Sam Gibson's. Should probably check with Lucy or Sam.
        poloidalFlux = np.linspace(0, 1.0, len(rBphi))
        poloidalFlux_grid = (unnorm_psi_2D - psi_axis) / (psi_boundary - psi_axis)

        return cls(
            R_grid=data_R_coord,
            Z_grid=data_Z_coord,
            rBphi=rBphi,
            psi_norm_2D=poloidalFlux_grid,
            psi_norm_1D=poloidalFlux,
            psi_unnorm_axis=psi_axis,
            psi_unnorm_boundary=psi_boundary,
            delta_R=delta_R,
            delta_Z=delta_Z,
            interp_order=interp_order,
            interp_smoothing=interp_smoothing,
        )

    @classmethod
    def from_MAST_U_saved(
        cls,
        filename: pathlib.Path,
        equil_time: float,
        delta_R: float,
        delta_Z: float,
        interp_order: int,
        interp_smoothing: int,
    ):
        with np.load(filename) as loadfile:
            time_EFIT = loadfile["time_EFIT"]
            t_idx = find_nearest(time_EFIT, equil_time)
            print("EFIT time", time_EFIT[t_idx])

            return EFITField(
                R_grid=loadfile["R_EFIT"],
                Z_grid=loadfile["Z_EFIT"],
                rBphi=loadfile["rBphi"][t_idx, :],
                psi_norm_2D=loadfile["poloidalFlux_grid"][t_idx, :, :],
                psi_norm_1D=loadfile["poloidalFlux"][t_idx, :],
                psi_unnorm_axis=loadfile["poloidalFlux_unnormalised_axis"][t_idx],
                psi_unnorm_boundary=loadfile["poloidalFlux_unnormalised_boundary"][
                    t_idx
                ],
                delta_R=delta_R,
                delta_Z=delta_Z,
                interp_order=interp_order,
                interp_smoothing=interp_smoothing,
            )
