# Copyright 2023 - 2023, Valerian Hall-Chen and the Scotty contributors
# SPDX-License-Identifier: GPL-3.0

from abc import ABC, abstractmethod
import inspect
import json
from multiprocessing.managers import ValueProxy
import numpy as np
import pathlib
import h5py
from scipy.interpolate import (
    RectBivariateSpline,
    RegularGridInterpolator,
    UnivariateSpline,
)
from scotty.derivatives import derivative
from scotty.fun_general import find_nearest, find_q_labframe_cart_to_cyl
from scotty.logger import logging, timer
from scotty.torbeam import Torbeam
from scotty.typing import ArrayLike, FloatArray, PathLike
from typing import cast, Callable, Literal, Optional, Tuple, Union, List

log = logging.getLogger()

##################################################
#
# SPLINE FUNCTIONS
#
##################################################


def get_profile_data(
    find_B_method: str,
    geometry: Literal["cylindrical", "cartesian"],
    magnetic_data_path: pathlib.Path,
) -> Tuple[
    FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray
]:
    """The grids, field and poloidal flux in the file ``magnetic_data_path``: a
    TORBEAM topfile (``"torbeam"``) or its OMFIT JSON form (``"omfit"``)"""

    ##############################################
    # OMFIT
    ##############################################
    if find_B_method == "omfit":
        log.info(f"Using OMFIT JSON Torbeam file for B and poloidal flux")

        def unflatten(R_coord, Z_coord, arr):
            """Convert from column-major (TORBEAM, Fortran) to row-major order (Scotty, Python)"""
            return np.asarray(arr).reshape(len(Z_coord), len(R_coord)).T

        if geometry == "cylindrical":
            with open(magnetic_data_path) as f:
                data = json.load(f)
            q0 = np.array(data["R"])
            q1 = np.array([0])
            q2 = np.array(data["Z"])
            B0 = unflatten(q0, q2, np.array(data["Br"]))
            B1 = unflatten(q0, q2, np.array(data["Bt"]))
            B2 = unflatten(q0, q2, np.array(data["Bz"]))
            psi = unflatten(q0, q2, np.array(data["pol_flux"]))
        else:  # geometry == "cartesian":
            raise ValueError(
                f"`find_B_method` = 'omfit' only works for `geometry` = 'cylindrical' for now"
            )

    ##############################################
    # TORBEAM
    ##############################################
    elif find_B_method == "torbeam":
        log.info(f"Using TORBEAM input files for B and poloidal flux")
        if geometry == "cylindrical":
            torbeam = Torbeam.from_file(magnetic_data_path)
            q0 = torbeam.R_grid
            q1 = np.array([0])
            q2 = torbeam.Z_grid
            B0 = torbeam.B_R
            B1 = torbeam.B_T
            B2 = torbeam.B_Z
            psi = torbeam.psi
        else:  # geometry == "cartesian":
            raise ValueError(
                f"`find_B_method` = 'torbeam' only works for `geometry` = 'cylindrical' for now"
            )

    else:
        raise ValueError(f"`find_B_method` = '{find_B_method}' is not supported")

    return q0, q1, q2, B0, B1, B2, psi


##################################################
#
# SPLINE FUNCTIONS
#
##################################################


@timer
def _make_rect_spline(
    R_coord, Z_coord, data_array, interp_order_int: int, interp_smoothing: int
) -> Tuple[
    Callable[[ArrayLike, ArrayLike, ArrayLike], FloatArray], RectBivariateSpline
]:
    spline = RectBivariateSpline(
        x=R_coord,
        y=Z_coord,
        z=data_array,
        bbox=[None, None, None, None],
        kx=interp_order_int,
        ky=interp_order_int,
        s=interp_smoothing,
    )
    return lambda q_R, _, q_Z: spline(q_R, q_Z, grid=False), spline


def _make_rect_spline_derivatives(
    spline: RectBivariateSpline,
) -> Tuple[
    Callable[[ArrayLike, ArrayLike, ArrayLike, float], FloatArray],
    Callable[[ArrayLike, ArrayLike, ArrayLike, float], FloatArray],
    Callable[[ArrayLike, ArrayLike, ArrayLike, float], FloatArray],
    Callable[[ArrayLike, ArrayLike, ArrayLike, float], FloatArray],
    Callable[[ArrayLike, ArrayLike, ArrayLike, float, float], FloatArray],
]:
    dpsi_dR = spline.partial_derivative(1, 0)
    dpsi_dZ = spline.partial_derivative(0, 1)
    d2psi_dR2 = spline.partial_derivative(2, 0)
    d2psi_dZ2 = spline.partial_derivative(0, 2)
    d2psi_dRdZ = spline.partial_derivative(1, 1)

    return (
        lambda q_R, _, q_Z, *args: dpsi_dR(q_R, q_Z, grid=False),
        lambda q_R, _, q_Z, *args: dpsi_dZ(q_R, q_Z, grid=False),
        lambda q_R, _, q_Z, *args: d2psi_dR2(q_R, q_Z, grid=False),
        lambda q_R, _, q_Z, *args: d2psi_dZ2(q_R, q_Z, grid=False),
        lambda q_R, _, q_Z, *args: d2psi_dRdZ(q_R, q_Z, grid=False),
    )


CUBOID_SPLINE_SOLVER_ARGS = {"rtol": 1e-12, "atol": 0.0}
"""Tolerances for the sparse solve that gives the coefficients of the 3D
``"slinear"``, ``"cubic"`` and ``"quintic"`` splines of
`InterpolatedField_Cartesian`. scipy solves for them iteratively with
`scipy.sparse.linalg.gcrotmk`, whose default stops once the residual is below
``rtol * ||b||`` with ``rtol = 1e-5`` and ``b`` all the grid values (so the
threshold grows with the size of the grid and the scale of the data). The
coefficient errors that leaves are amplified by the second derivatives of the
poloidal flux, to several percent for a tokamak field on a 2-4 cm grid.
``atol`` must be 0 too: scipy otherwise sets it to 1e-6. If the solve does not
reach these tolerances, `_make_cuboid_spline` raises an error rather than
returning a poor spline."""

# scipy >= 1.13 builds these splines by a sparse solve and takes `solver_args`;
# older versions build them recursively, with nothing to set
_RGI_TAKES_SOLVER_ARGS = (
    "solver_args" in inspect.signature(RegularGridInterpolator).parameters
)


@timer
def _make_cuboid_spline(
    X_coord, Y_coord, Z_coord, data_array, interp_order_str: str
) -> Tuple[
    Callable[[ArrayLike, ArrayLike, ArrayLike], FloatArray], RegularGridInterpolator
]:
    solver = {}
    if _RGI_TAKES_SOLVER_ARGS and interp_order_str in ("slinear", "cubic", "quintic"):
        solver["solver_args"] = dict(CUBOID_SPLINE_SOLVER_ARGS)
    try:
        spline = RegularGridInterpolator(
            points=(X_coord, Y_coord, Z_coord),
            values=data_array,
            method=interp_order_str,
            bounds_error=False,
            **solver,
        )
    except ValueError as error:
        if not solver:
            raise
        raise ValueError(
            "The sparse solve for the 3D spline coefficients did not reach the "
            f"tolerances in `geometry.CUBOID_SPLINE_SOLVER_ARGS` "
            f"({CUBOID_SPLINE_SOLVER_ARGS}). Loosening them makes the second "
            "derivatives of the poloidal flux inaccurate; a smoother or coarser "
            "grid may converge instead"
        ) from error
    return lambda X, Y, Z: spline((X, Y, Z)), spline


##################################################
#
# FIELD CLASSES, CYLINDRICAL
#
##################################################


class MagneticField_Cylindrical(ABC):
    """Abstract base class for cylindrical magnetic field geometries.
    Child classes must implement these methods"""

    R_coord: FloatArray  #: Sample locations for the major radius coordinate
    Z_coord: FloatArray  #: Sample locations for the vertical coordinate
    poloidalFlux_grid: FloatArray  #: Value of the poloidal magnetic flux, :math:`\psi`, on ``(R_coord, Z_coord)``
    # TODO: Include B_R grid, B_T grid, B_Z grids

    #: The file the field was read from, by a constructor such as
    #: `EFITField.from_MAST_saved`; saved in the output as the input
    #: ``magnetic_data_path``. None for a field not read from a file
    data_file: Optional[str] = None
    #: The time of the field in ``data_file`` (in the file's units): the time
    #: in the file nearest the one asked for. Saved in the output as the input
    #: ``magnetic_data_time``. None for a file with one time, or a field not
    #: read from a file
    data_time: Optional[float] = None

    @abstractmethod
    def B_R(self, R: ArrayLike, _: ArrayLike, Z: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    @abstractmethod
    def B_T(self, R: ArrayLike, _: ArrayLike, Z: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    @abstractmethod
    def B_Z(self, R: ArrayLike, _: ArrayLike, Z: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    @abstractmethod
    def polflux(self, R: ArrayLike, _: ArrayLike, Z: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    @abstractmethod
    def d_polflux_dR(
        self, R: ArrayLike, _: ArrayLike, Z: ArrayLike, delta_R: float
    ) -> ArrayLike:
        raise NotImplementedError

    @abstractmethod
    def d_polflux_dZ(
        self, R: ArrayLike, _: ArrayLike, Z: ArrayLike, delta_Z: float
    ) -> ArrayLike:
        raise NotImplementedError

    @abstractmethod
    def d2_polflux_dR2(
        self, R: ArrayLike, _: ArrayLike, Z: ArrayLike, delta_R: float
    ) -> ArrayLike:
        raise NotImplementedError

    @abstractmethod
    def d2_polflux_dZ2(
        self, R: ArrayLike, _: ArrayLike, Z: ArrayLike, delta_Z: float
    ) -> ArrayLike:
        raise NotImplementedError

    @abstractmethod
    def d2_polflux_dRdZ(
        self, R: ArrayLike, _: ArrayLike, Z: ArrayLike, delta_R: float, delta_Z: float
    ) -> ArrayLike:
        raise NotImplementedError

    def _calculate_incyl_outcyl(
        self,
        R: ArrayLike,
        _: ArrayLike,
        Z: ArrayLike,
        vector: bool = False,
        unitvector: bool = False,
        magnitude: bool = False,
    ) -> List[FloatArray]:
        toreturn = []
        B_R = self.B_R(R, _, Z)
        B_T = self.B_T(R, _, Z)
        B_Z = self.B_Z(R, _, Z)
        B_vec = np.array([B_R, B_T, B_Z])

        if vector:
            toreturn.append(B_vec.T)
        if unitvector or magnitude:
            B_mag = np.sqrt(B_R**2 + B_T**2 + B_Z**2)
            if unitvector:
                b_hat = B_vec / B_mag
                toreturn.append(b_hat.T)
            if magnitude:
                toreturn.append(B_mag)

        return toreturn

    def all(self, R: ArrayLike, _: ArrayLike, Z: ArrayLike) -> List[FloatArray]:
        r"""Returns, in this order: :math:`\mathbf{B}`, :math:`\mathbf{B}/|B|`, and :math:`|B|`,
        which are the the vector, unit vector, and magnitude of the magnetic field respectively

        Input: cylindrical (R, zeta, Z)

        Output: cylindrical
        """
        return self._calculate_incyl_outcyl(
            R, _, Z, vector=True, unitvector=True, magnitude=True
        )

    def vector(self, R: ArrayLike, _: ArrayLike, Z: ArrayLike) -> FloatArray:
        r"""Returns :math:`\mathbf{B}`, the vector of the magnetic field

        Input: cylindrical (R, zeta, Z)

        Output: cylindrical (B_R, B_T, B_Z)
        """
        return self._calculate_incyl_outcyl(
            R, _, Z, vector=True, unitvector=False, magnitude=False
        )[0]

    def unitvector(self, R: ArrayLike, _: ArrayLike, Z: ArrayLike) -> FloatArray:
        r"""Returns :math:`\mathbf{B}/|B|`, the unit vector of the magnetic field

        Input: cylindrical (R, zeta, Z)

        Output: cylindrical (b_R, b_T, b_Z)
        """
        return self._calculate_incyl_outcyl(
            R, _, Z, vector=False, unitvector=True, magnitude=False
        )[0]

    def magnitude(self, R: ArrayLike, _: ArrayLike, Z: ArrayLike) -> ArrayLike:
        r"""Returns :math:`|B|`, the magnitude of the magnetic field

        Input: cylindrical (R, zeta, Z)
        """
        return self._calculate_incyl_outcyl(
            R, _, Z, vector=False, unitvector=False, magnitude=True
        )[0]

    # For abstraction purposes
    def B_X(self, R: ArrayLike, zeta: ArrayLike, Z: ArrayLike) -> ArrayLike:
        return self.B_R(R, zeta, Z) * np.cos(zeta) - self.B_T(R, zeta, Z) * np.sin(zeta)

    def B_Y(self, R: ArrayLike, zeta: ArrayLike, Z: ArrayLike) -> ArrayLike:
        return self.B_R(R, zeta, Z) * np.sin(zeta) + self.B_T(R, zeta, Z) * np.cos(zeta)

    def unitvector_cartesian(
        self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike
    ) -> FloatArray:
        r"""Returns :math:`\mathbf{B}/|B|`, the unit vector of the magnetic field"""
        R, zeta, Z = find_q_labframe_cart_to_cyl(np.array([X, Y, Z]))
        magnitude = self.magnitude(R, zeta, Z)
        vector = np.array(
            [self.B_X(R, zeta, Z), self.B_Y(R, zeta, Z), self.B_Z(R, zeta, Z)]
        )
        unitvector = cast(FloatArray, (vector / magnitude).T)
        return unitvector

    def polflux_cartesian(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> ArrayLike:
        return self.polflux(np.sqrt(X**2 + Y**2), 0, Z)

    def d_polflux_dX(
        self, R: ArrayLike, zeta: ArrayLike, Z: ArrayLike, delta_R: float
    ) -> ArrayLike:
        return np.cos(zeta) * self.d_polflux_dR(R, zeta, Z, delta_R)

    def d_polflux_dY(
        self, R: ArrayLike, zeta: ArrayLike, Z: ArrayLike, delta_R: float
    ) -> ArrayLike:
        return np.sin(zeta) * self.d_polflux_dR(R, zeta, Z, delta_R)


def _record_source(field, filename: PathLike, time: Optional[float]):
    """Set ``field.data_file`` and ``field.data_time`` (the time used, or None
    for a file with one time), so that the output records them; returns
    ``field``"""
    field.data_file = str(pathlib.Path(filename))
    field.data_time = None if time is None else float(time)
    return field


def _time_index(
    loadfile, time_key: str, equil_time: Optional[float], filename: PathLike
):
    """Index of the time nearest ``equil_time`` in ``loadfile[time_key]``,
    or ``...`` (the whole array) for a file with one time and no
    ``time_key``, which takes ``equil_time=None``"""
    if time_key not in loadfile:
        if equil_time is not None:
            raise ValueError(
                f"`equil_time` = {equil_time} was given, but '{filename}' has "
                f"one time (no '{time_key}'): leave `equil_time` out"
            )
        return ...
    if equil_time is None:
        raise ValueError(
            f"'{filename}' has several times ('{time_key}'): give `equil_time`"
        )
    times = loadfile[time_key]
    index = find_nearest(times, equil_time)
    log.info(f"Field at time {times[index]} (nearest to {equil_time})")
    return index


class InterpolatedField_Cylindrical(MagneticField_Cylindrical):
    r"""Field interpolated from grids of :math:`B_R`, :math:`B_T`,
    :math:`B_Z` and the normalised poloidal flux on an (R, Z) grid. Give it
    to `beam_me_up` as ``find_B_method``. `from_npz` reads one from a
    ``.npz`` file"""

    def __init__(
        self,
        R_coord: FloatArray,
        Z_coord: FloatArray,
        B_R: FloatArray,
        B_T: FloatArray,
        B_Z: FloatArray,
        psi: FloatArray,
        interp_order: int = 5,
        interp_smoothing: int = 0,
    ):

        # Defining class attributes for the interpolated grids
        self.R_coord = R_coord
        self.Z_coord = Z_coord
        self.B_R_grid = B_R
        self.B_T_grid = B_T
        self.B_Z_grid = B_Z
        self.poloidalFlux_grid = psi
        self.interp_order = interp_order

        (self._interp_B_R, self._spline_B_R), duration_B_R_interpolation = (
            _make_rect_spline(R_coord, Z_coord, B_R, interp_order, interp_smoothing)
        )
        log.debug(f"Interpolating 2D B_R profile took {duration_B_R_interpolation} s")

        (self._interp_B_T, self._spline_B_T), duration_B_T_interpolation = (
            _make_rect_spline(R_coord, Z_coord, B_T, interp_order, interp_smoothing)
        )
        log.debug(f"Interpolating 2D B_T profile took {duration_B_T_interpolation} s")

        (self._interp_B_Z, self._spline_B_Z), duration_B_Z_interpolation = (
            _make_rect_spline(R_coord, Z_coord, B_Z, interp_order, interp_smoothing)
        )
        log.debug(f"Interpolating 2D B_Z profile took {duration_B_Z_interpolation} s")

        (self._interp_polflux, self._spline_polflux), duration_psi_interpolation = (
            _make_rect_spline(R_coord, Z_coord, psi, interp_order, interp_smoothing)
        )
        log.debug(
            f"Interpolating 2D poloidal flux profile took {duration_psi_interpolation} s"
        )

        self._set_poloidal_flux_derivatives(self._spline_polflux)

    @classmethod
    def from_npz(
        cls,
        filename: PathLike,
        equil_time: Optional[float] = None,
        interp_order: int = 5,
        interp_smoothing: int = 0,
    ) -> "InterpolatedField_Cylindrical":
        """The field in a ``.npz`` file with grids of B and the normalised
        poloidal flux: ``R_EFIT`` and ``Z_EFIT`` (the R and Z coordinates),
        and ``Br_grid``, ``Bphi_grid``, ``Bz_grid`` and ``poloidalFlux_grid``
        on them, for example a MAST-U file saved from UDA.

        A file with several times also has ``time_EFIT``, and the grids have
        time as their first axis: give ``equil_time`` (in the file's units),
        and the field is taken at the nearest time. A file with one time has
        no ``time_EFIT``: leave ``equil_time`` out. These were v3's
        ``find_B_method`` ``"test"`` and ``"test_notime"``"""
        with np.load(filename) as loadfile:
            index = _time_index(loadfile, "time_EFIT", equil_time, filename)
            return _record_source(
                cls(
                    R_coord=loadfile["R_EFIT"],
                    Z_coord=loadfile["Z_EFIT"],
                    B_R=loadfile["Br_grid"][index],
                    B_T=loadfile["Bphi_grid"][index],
                    B_Z=loadfile["Bz_grid"][index],
                    psi=loadfile["poloidalFlux_grid"][index],
                    interp_order=interp_order,
                    interp_smoothing=interp_smoothing,
                ),
                filename,
                None if index is ... else loadfile["time_EFIT"][index],
            )

    def _set_poloidal_flux_derivatives(self, psi_spline):
        try:
            (
                self._dpsi_dR,
                self._dpsi_dZ,
                self._d2psi_dR2,
                self._d2psi_dZ2,
                self._d2psi_dRdZ,
            ) = _make_rect_spline_derivatives(psi_spline)

        except AttributeError:
            # Older versions of SciPy don't have
            # `RectBivariateSpline.partial_derivative`, so fall back
            # to base class implementation of flux derivatives
            self._dpsi_dR = super().d_polflux_dR
            self._dpsi_dZ = super().d_polflux_dZ
            self._d2psi_dR2 = super().d2_polflux_dR2
            self._d2psi_dZ2 = super().d2_polflux_dZ2
            self._d2psi_dRdZ = super().d2_polflux_dRdZ

    # Defining class attributes for B_R, B_T, B_Z, and polflux
    def B_R(self, R: ArrayLike, _: ArrayLike, Z: ArrayLike) -> ArrayLike:
        return self._interp_B_R(R, _, Z)

    def B_T(self, R: ArrayLike, _: ArrayLike, Z: ArrayLike) -> ArrayLike:
        return self._interp_B_T(R, _, Z)

    def B_Z(self, R: ArrayLike, _: ArrayLike, Z: ArrayLike) -> ArrayLike:
        return self._interp_B_Z(R, _, Z)

    def polflux(self, R: ArrayLike, _: ArrayLike, Z: ArrayLike) -> ArrayLike:
        return self._interp_polflux(R, _, Z)

    # Defining the class attributes for the first- and second-order derivatives of polflux
    def d_polflux_dR(
        self, R: ArrayLike, _: ArrayLike, Z: ArrayLike, delta_R: float
    ) -> ArrayLike:
        return self._dpsi_dR(R, _, Z, delta_R)

    def d_polflux_dZ(
        self, R: ArrayLike, _: ArrayLike, Z: ArrayLike, delta_Z: float
    ) -> ArrayLike:
        return self._dpsi_dZ(R, _, Z, delta_Z)

    def d2_polflux_dR2(
        self, R: ArrayLike, _: ArrayLike, Z: ArrayLike, delta_R: float
    ) -> ArrayLike:
        return self._d2psi_dR2(R, _, Z, delta_R)

    def d2_polflux_dZ2(
        self, R: ArrayLike, _: ArrayLike, Z: ArrayLike, delta_Z: float
    ) -> ArrayLike:
        return self._d2psi_dZ2(R, _, Z, delta_Z)

    def d2_polflux_dRdZ(
        self, R: ArrayLike, _: ArrayLike, Z: ArrayLike, delta_R: float, delta_Z: float
    ) -> ArrayLike:
        return self._d2psi_dRdZ(R, _, Z, delta_R, delta_Z)


class EFITField(InterpolatedField_Cylindrical):
    r"""Field from EFIT output: the normalised poloidal flux on an (R, Z) grid
    and :math:`R B_\phi` as a function of normalised flux. :math:`B_R` and
    :math:`B_Z` are computed from the flux gradient, and :math:`B_T` from
    :math:`R B_\phi`, rather than interpolated from grids

    Make one from a file with `from_MAST_saved`, `from_MAST_U_saved` or
    `from_EFITpp`, and give it to `beam_me_up` as ``find_B_method``. They
    replace v3's ``find_B_method`` ``"UDA_saved"`` and ``"EFITpp"``: the file
    and the time are their arguments, rather than ``magnetic_data_path``,
    ``shot`` and ``equil_time``. For a field interpolated from saved grids
    of B, see `InterpolatedField_Cylindrical.from_npz`
    """

    def __init__(
        self,
        R_grid: FloatArray,
        Z_grid: FloatArray,
        rBphi: FloatArray,
        psi_norm_2D: FloatArray,
        psi_unnorm_axis: float,
        psi_unnorm_boundary: float,
        psi_norm_1D: Optional[FloatArray] = None,
        delta_R: Optional[float] = 0.0001,
        delta_Z: Optional[float] = 0.0001,
        interp_order: int = 5,
        interp_smoothing: int = 0,
    ):
        self.R_coord = R_grid
        self.Z_coord = Z_grid
        self.poloidalFlux_grid = psi_norm_2D
        self.interp_order = interp_order

        self.delta_R = delta_R or 1e-8
        self.delta_Z = delta_Z or 1e-8

        (self._interp_polflux, self._spline_polflux), _ = _make_rect_spline(
            R_grid, Z_grid, psi_norm_2D, interp_order, interp_smoothing
        )
        self._set_poloidal_flux_derivatives(self._spline_polflux)

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

    def B_R(self, R: ArrayLike, _: ArrayLike, Z: ArrayLike) -> ArrayLike:
        dpolflux_dZ = self.d_polflux_dZ(R, _, Z, self.delta_Z)
        return -dpolflux_dZ * self.poloidal_flux_gradient / np.asarray(R)

    def B_T(self, R: ArrayLike, _: ArrayLike, Z: ArrayLike) -> ArrayLike:
        polflux = self.polflux(R, _, Z)
        return self._interp_rBphi(polflux) / np.asarray(R)

    def B_Z(self, R: ArrayLike, _: ArrayLike, Z: ArrayLike) -> ArrayLike:
        dpolflux_dR = self.d_polflux_dR(R, _, Z, self.delta_R)
        return dpolflux_dR * self.poloidal_flux_gradient / np.asarray(R)

    @classmethod
    def from_EFITpp(
        cls,
        filename: PathLike,
        equil_time: float,
        delta_R: Optional[float] = 0.0001,
        delta_Z: Optional[float] = 0.0001,
        interp_order: int = 5,
        interp_smoothing: int = 0,
    ) -> "EFITField":
        """The field at the time nearest ``equil_time`` (in the file's
        units) in an EFIT++ output file (``efitOut.nc``, for example the
        MSE-constrained EFIT++ of MAST). v3's ``find_B_method="EFITpp"``"""
        # Read with h5py: h5netcdf's legacy API (1.8.x) builds variable paths
        # from the group's short name, so it cannot open variables two groups
        # deep, such as output/profiles2D/poloidalFlux
        with h5py.File(filename, "r") as efitpp:
            efitpp_times = efitpp["time"][:]
            time_idx = find_nearest(efitpp_times, equil_time)
            log.info(f"EFIT++ time {efitpp_times[time_idx]}")

            profiles2D = efitpp["output/profiles2D"]
            # unnormalised, as a function of R and Z
            unnorm_psi_2D = profiles2D["poloidalFlux"][time_idx]
            data_R_coord = profiles2D["r"][time_idx]
            data_Z_coord = profiles2D["z"][time_idx]

            fluxFunctionProfiles = efitpp["output/fluxFunctionProfiles"]
            norm_psi_1D = fluxFunctionProfiles["normalizedPoloidalFlux"][:]
            # poloidalFlux as a function of normalised poloidal flux
            unorm_psi_1D = fluxFunctionProfiles["poloidalFlux"][time_idx]
            rBphi = fluxFunctionProfiles["rBphi"][time_idx]

        # linear fit
        # psi_norm = m psi + c, so psi is -c/m on axis and (1 - c)/m at the
        # boundary: EFITField needs psi_boundary - psi_axis = dpsi/dpsi_norm =
        # 1/m (passing m made B_p wrong by a factor m^2)
        polflux_const_m, polflux_const_c = np.polyfit(unorm_psi_1D, norm_psi_1D, 1)
        norm_psi_2D = unnorm_psi_2D * polflux_const_m + polflux_const_c

        return _record_source(
            cls(
                R_grid=data_R_coord,
                Z_grid=data_Z_coord,
                rBphi=rBphi,
                psi_norm_2D=norm_psi_2D,
                psi_norm_1D=norm_psi_1D,
                psi_unnorm_axis=-polflux_const_c / polflux_const_m,
                psi_unnorm_boundary=(1 - polflux_const_c) / polflux_const_m,
                delta_R=delta_R,
                delta_Z=delta_Z,
                interp_order=interp_order,
                interp_smoothing=interp_smoothing,
            ),
            filename,
            efitpp_times[time_idx],
        )

    @classmethod
    def from_MAST_saved(
        cls,
        filename: PathLike,
        equil_time: float,
        delta_R: Optional[float] = 0.0001,
        delta_Z: Optional[float] = 0.0001,
        interp_order: int = 5,
        interp_smoothing: int = 0,
    ) -> "EFITField":
        """The field at the time nearest ``equil_time`` (in the file's
        units) in a MAST EFIT file saved from UDA (``.npz``, as
        ``{shot}_equilibrium_data.npz``, with time bases ``t_base_B`` and
        ``t_base_C``). v3's ``find_B_method="UDA_saved"`` for MAST shots
        (up to 30471)"""
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
        log.info(f"EFIT time {t_base_B[t_base_B_idx]}")

        rBphi = rBphi_all_times[t_base_C_idx, :]
        psi_axis = polflux_axis_all_times[t_base_C_idx]
        psi_boundary = psi_boundary_all_times[t_base_C_idx]
        unnorm_psi_2D = unnorm_psi_all_times[t_base_B_idx, :, :].T

        # Taken from an old file of Sam Gibson's. Should probably check with Lucy or Sam.
        poloidalFlux = np.linspace(0, 1.0, len(rBphi))
        poloidalFlux_grid = (unnorm_psi_2D - psi_axis) / (psi_boundary - psi_axis)

        return _record_source(
            cls(
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
            ),
            filename,
            t_base_B[t_base_B_idx],
        )

    @classmethod
    def from_MAST_U_saved(
        cls,
        filename: PathLike,
        equil_time: Optional[float] = None,
        delta_R: Optional[float] = 0.0001,
        delta_Z: Optional[float] = 0.0001,
        interp_order: int = 5,
        interp_smoothing: int = 0,
    ) -> "EFITField":
        """The field in a MAST-U EFIT file saved from UDA (``.npz``, as
        ``{shot}_equilibrium_data.npz``): at the time nearest ``equil_time``
        (in the file's units, ``time_EFIT``), or, for a file with one time
        and no ``time_EFIT``, ``equil_time=None``. v3's
        ``find_B_method="UDA_saved"`` for MAST-U shots, and with no shot.

        Without ``time_EFIT``, the arrays have no time axis.

        ``rBphi`` is on a normalised flux evenly spaced from 0 to 1. The
        file's ``poloidalFlux`` is not that flux: in the MAST-U files saved
        from UDA, it goes from 1 to 0 and back to 1. v3 used it for
        ``rBphi`` with several times, which failed on those files ("x must
        be strictly increasing"). On shot 45290 at 0.15 s, with ``rBphi`` on
        the evenly spaced flux, B_T agrees with the file's ``Bphi_grid`` to
        3e-5 of its largest value inside the plasma (normalised flux below
        0.9)"""
        with np.load(filename) as loadfile:
            index = _time_index(loadfile, "time_EFIT", equil_time, filename)

            return _record_source(
                cls(
                    R_grid=loadfile["R_EFIT"],
                    Z_grid=loadfile["Z_EFIT"],
                    rBphi=loadfile["rBphi"][index],
                    psi_norm_2D=loadfile["poloidalFlux_grid"][index],
                    psi_unnorm_axis=loadfile["poloidalFlux_unnormalised_axis"][index],
                    psi_unnorm_boundary=loadfile["poloidalFlux_unnormalised_boundary"][
                        index
                    ],
                    delta_R=delta_R,
                    delta_Z=delta_Z,
                    interp_order=interp_order,
                    interp_smoothing=interp_smoothing,
                ),
                filename,
                None if index is ... else loadfile["time_EFIT"][index],
            )


class CircularFieldBase(MagneticField_Cylindrical):
    r"""Base class for analytical geometries with circular flux surfaces,
    where the poloidal flux is :math:`\psi = \rho / a`, with
    :math:`\rho = \sqrt{(R - R_{axis})^2 + Z^2}` and :math:`a` the minor radius.

    Child classes must set ``R_axis`` and ``minor_radius_a``, and implement
    ``B_R``, ``B_T`` and ``B_Z``. A child class with a different flux function
    overrides ``polflux`` and its analytic derivatives too (see
    `CircularFluxSurfaceField`).

    Called ``CircularFluxSurfaceField`` before 2026-10-30; that name now
    belongs to the divergence-free field of beam-tracing/Scotty#161, as in v3.

    The poloidal flux derivatives are exact (``derivative_method =
    "analytic"``) by default, or can be taken by finite differences of
    ``polflux`` (``derivative_method = "numerical"``) with the step sizes
    passed to each method. All of them are singular on the magnetic axis
    (:math:`\rho = 0`).
    """

    R_axis: float
    minor_radius_a: float
    derivative_method: Literal["analytic", "numerical"] = "analytic"

    def _set_derivative_method(
        self, derivative_method: Literal["analytic", "numerical"]
    ) -> None:
        if derivative_method not in ["analytic", "numerical"]:
            raise ValueError(
                f"`derivative_method` must be 'analytic' or 'numerical', but got '{derivative_method}'"
            )
        self.derivative_method = derivative_method

    def rho(self, R: ArrayLike, _: ArrayLike, Z: ArrayLike) -> FloatArray:
        R, Z = np.asarray(R, dtype=np.float64), np.asarray(Z, dtype=np.float64)
        return np.sqrt((R - self.R_axis) ** 2 + Z**2)

    def polflux(self, R: ArrayLike, _: ArrayLike, Z: ArrayLike) -> FloatArray:
        R, Z = np.asarray(R, dtype=np.float64), np.asarray(Z, dtype=np.float64)
        return self.rho(R, _, Z) / self.minor_radius_a

    def _polflux_RZ(self, R: ArrayLike, Z: ArrayLike) -> FloatArray:
        # Axisymmetric, so `derivative` only needs R and Z
        return self.polflux(R, 0.0, Z)

    def _numerical_derivative(
        self, dims: Union[str, Tuple[str, str]], R: ArrayLike, Z: ArrayLike, spacings
    ) -> FloatArray:
        return derivative(self._polflux_RZ, dims, {"R": R, "Z": Z}, spacings)

    def d_polflux_dR(
        self, R: ArrayLike, _: ArrayLike, Z: ArrayLike, delta_R: float
    ) -> FloatArray:
        if self.derivative_method == "numerical":
            return self._numerical_derivative("R", R, Z, {"R": delta_R})
        R, Z = np.asarray(R, dtype=np.float64), np.asarray(Z, dtype=np.float64)
        return (R - self.R_axis) / (self.minor_radius_a * self.rho(R, _, Z))

    def d_polflux_dZ(
        self, R: ArrayLike, _: ArrayLike, Z: ArrayLike, delta_Z: float
    ) -> FloatArray:
        if self.derivative_method == "numerical":
            return self._numerical_derivative("Z", R, Z, {"Z": delta_Z})
        R, Z = np.asarray(R, dtype=np.float64), np.asarray(Z, dtype=np.float64)
        return Z / (self.minor_radius_a * self.rho(R, _, Z))

    def d2_polflux_dR2(
        self, R: ArrayLike, _: ArrayLike, Z: ArrayLike, delta_R: float
    ) -> FloatArray:
        if self.derivative_method == "numerical":
            return self._numerical_derivative(("R", "R"), R, Z, {"R": delta_R})
        R, Z = np.asarray(R, dtype=np.float64), np.asarray(Z, dtype=np.float64)
        return Z**2 / (self.minor_radius_a * self.rho(R, _, Z) ** 3)

    def d2_polflux_dZ2(
        self, R: ArrayLike, _: ArrayLike, Z: ArrayLike, delta_Z: float
    ) -> FloatArray:
        if self.derivative_method == "numerical":
            return self._numerical_derivative(("Z", "Z"), R, Z, {"Z": delta_Z})
        R, Z = np.asarray(R, dtype=np.float64), np.asarray(Z, dtype=np.float64)
        return (R - self.R_axis) ** 2 / (self.minor_radius_a * self.rho(R, _, Z) ** 3)

    def d2_polflux_dRdZ(
        self, R: ArrayLike, _: ArrayLike, Z: ArrayLike, delta_R: float, delta_Z: float
    ) -> FloatArray:
        if self.derivative_method == "numerical":
            return self._numerical_derivative(
                ("R", "Z"), R, Z, {"R": delta_R, "Z": delta_Z}
            )
        R, Z = np.asarray(R, dtype=np.float64), np.asarray(Z, dtype=np.float64)
        return -(R - self.R_axis) * Z / (self.minor_radius_a * self.rho(R, _, Z) ** 3)


class CircularFluxSurfaceField(CircularFieldBase):
    r"""Circular flux surfaces with a poloidal field that increases linearly in minor radius

    .. math::

        \psi_{pol} = \left(\frac{r}{a}\right)^2, \qquad
        B_p = B_{p,a} \frac{r}{a} \frac{R_{axis}}{R}, \qquad
        B_T = B_{T,axis} \frac{R_{axis}}{R},

    where :math:`r = \sqrt{(R - R_{axis})^2 + Z^2}` is the minor radius, so
    :math:`\rho_{pol} = r/a`. The poloidal field is the gradient of the
    dimensional flux :math:`\Psi = B_{p,a} a R_{axis} \psi_{pol} / 2` crossed with
    :math:`\nabla \zeta`, so it is divergence-free and tangent to the flux surfaces.
    In the large-aspect-ratio limit this is the field of a constant current
    density; at finite aspect ratio the toroidal current density varies as
    :math:`(1 + R_{axis}/R)/R`.

    This is the field of beam-tracing/Scotty#161 (v3), ported so that
    ``find_B_method="analytical"`` describes the same plasma in v3 and v4, with
    the same signs of :math:`B_R` and :math:`B_Z`. Compared with Sec. 2.4 of
    Hall-Chen, Parra & Hillesheim, PPCF 64, 095002 (2022), it adds the
    :math:`R_{axis}/R` factor in :math:`B_p`, fixes the sign of :math:`B_R, B_Z`
    (printed eqs. 81-82 are not tangent to the flux surfaces) and uses
    :math:`\psi_{pol} = (r/a)^2`.

    Unlike the base class's :math:`\psi = \rho / a`, its flux derivatives are smooth on the
    magnetic axis.

    Parameters
    ----------
    B_T_axis : float
        Toroidal magnetic field at the magnetic axis (Tesla)
    R_axis : float
        Major radius of the magnetic axis (metres)
    minor_radius_a : float
        Minor radius of the last closed flux surface (metres)
    B_p_a : float
        Poloidal magnetic field on the last closed flux surface at
        ``R = R_axis`` (Tesla)
    R_points : int
    Z_points : int
        Number of points for sample ``(R, Z)`` grid
    grid_buffer_factor : float
        Multiplicative factor to increase size of sample grid by
    derivative_method : str
        ``"analytic"`` (default) for exact poloidal flux derivatives, or
        ``"numerical"`` for finite differences
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
        derivative_method: Literal["analytic", "numerical"] = "analytic",
    ):
        self.B_T_axis = B_T_axis
        self.R_axis = R_axis
        self.minor_radius_a = minor_radius_a
        self.B_p_a = B_p_a
        self._set_derivative_method(derivative_method)

        grid_width = grid_buffer_factor * minor_radius_a
        self.R_coord = np.linspace(R_axis - grid_width, R_axis + grid_width, R_points)
        self.Z_coord = np.linspace(-grid_width, grid_width, Z_points)
        R_meshgrid, Z_meshgrid = np.meshgrid(self.R_coord, self.Z_coord, indexing="ij")
        self.poloidalFlux_grid = self.polflux(R_meshgrid, 0.0, Z_meshgrid)

    def B_R(self, R: ArrayLike, _: ArrayLike, Z: ArrayLike) -> FloatArray:
        R, Z = np.asarray(R, dtype=np.float64), np.asarray(Z, dtype=np.float64)
        return self.B_p_a * self.R_axis * Z / (R * self.minor_radius_a)

    def B_T(self, R: ArrayLike, _: ArrayLike, Z: ArrayLike) -> FloatArray:
        R, Z = np.asarray(R, dtype=np.float64), np.asarray(Z, dtype=np.float64)
        return self.B_T_axis * (self.R_axis / R) * np.ones_like(Z)

    def B_Z(self, R: ArrayLike, _: ArrayLike, Z: ArrayLike) -> FloatArray:
        R, Z = np.asarray(R, dtype=np.float64), np.asarray(Z, dtype=np.float64)
        return -self.B_p_a * self.R_axis * (R - self.R_axis) / (R * self.minor_radius_a)

    def polflux(self, R: ArrayLike, _: ArrayLike, Z: ArrayLike) -> FloatArray:
        R, Z = np.asarray(R, dtype=np.float64), np.asarray(Z, dtype=np.float64)
        return ((R - self.R_axis) ** 2 + Z**2) / self.minor_radius_a**2

    def d_polflux_dR(
        self, R: ArrayLike, _: ArrayLike, Z: ArrayLike, delta_R: float
    ) -> FloatArray:
        if self.derivative_method == "numerical":
            return self._numerical_derivative("R", R, Z, {"R": delta_R})
        R, Z = np.asarray(R, dtype=np.float64), np.asarray(Z, dtype=np.float64)
        return 2 * (R - self.R_axis) / self.minor_radius_a**2 * np.ones_like(Z)

    def d_polflux_dZ(
        self, R: ArrayLike, _: ArrayLike, Z: ArrayLike, delta_Z: float
    ) -> FloatArray:
        if self.derivative_method == "numerical":
            return self._numerical_derivative("Z", R, Z, {"Z": delta_Z})
        R, Z = np.asarray(R, dtype=np.float64), np.asarray(Z, dtype=np.float64)
        return 2 * Z / self.minor_radius_a**2 * np.ones_like(R)

    def d2_polflux_dR2(
        self, R: ArrayLike, _: ArrayLike, Z: ArrayLike, delta_R: float
    ) -> FloatArray:
        if self.derivative_method == "numerical":
            return self._numerical_derivative(("R", "R"), R, Z, {"R": delta_R})
        R, Z = np.asarray(R, dtype=np.float64), np.asarray(Z, dtype=np.float64)
        return np.full(np.broadcast(R, Z).shape, 2 / self.minor_radius_a**2)

    def d2_polflux_dZ2(
        self, R: ArrayLike, _: ArrayLike, Z: ArrayLike, delta_Z: float
    ) -> FloatArray:
        if self.derivative_method == "numerical":
            return self._numerical_derivative(("Z", "Z"), R, Z, {"Z": delta_Z})
        R, Z = np.asarray(R, dtype=np.float64), np.asarray(Z, dtype=np.float64)
        return np.full(np.broadcast(R, Z).shape, 2 / self.minor_radius_a**2)

    def d2_polflux_dRdZ(
        self, R: ArrayLike, _: ArrayLike, Z: ArrayLike, delta_R: float, delta_Z: float
    ) -> FloatArray:
        if self.derivative_method == "numerical":
            return self._numerical_derivative(
                ("R", "Z"), R, Z, {"R": delta_R, "Z": delta_Z}
            )
        R, Z = np.asarray(R, dtype=np.float64), np.asarray(Z, dtype=np.float64)
        return np.zeros(np.broadcast(R, Z).shape)


##################################################
#
# FIELD CLASSES, CARTESIAN
#
##################################################


class MagneticField_Cartesian(ABC):
    """Abstract base class for cartesian magnetic field geometries.
    Child classes must implement these methods"""

    X_coord: FloatArray
    Y_coord: FloatArray
    Z_coord: FloatArray
    # TODO: Include B_X, B_Y, B_Z, psi grids

    @abstractmethod
    def B_X(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    @abstractmethod
    def B_Y(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    @abstractmethod
    def B_Z(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    @abstractmethod
    def polflux(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    @abstractmethod
    def d_polflux_dX(
        self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_X: float
    ) -> ArrayLike:
        raise NotImplementedError

    @abstractmethod
    def d_polflux_dY(
        self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_Y: float
    ) -> ArrayLike:
        raise NotImplementedError

    @abstractmethod
    def d_polflux_dZ(
        self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_Z: float
    ) -> ArrayLike:
        raise NotImplementedError

    @abstractmethod
    def d2_polflux_dX2(
        self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_X: float
    ) -> ArrayLike:
        raise NotImplementedError

    @abstractmethod
    def d2_polflux_dY2(
        self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_Y: float
    ) -> ArrayLike:
        raise NotImplementedError

    @abstractmethod
    def d2_polflux_dZ2(
        self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_Z: float
    ) -> ArrayLike:
        raise NotImplementedError

    @abstractmethod
    def d2_polflux_dXdY(
        self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_X: float, delta_Y: float
    ) -> ArrayLike:
        raise NotImplementedError

    @abstractmethod
    def d2_polflux_dXdZ(
        self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_X: float, delta_Z: float
    ) -> ArrayLike:
        raise NotImplementedError

    @abstractmethod
    def d2_polflux_dYdZ(
        self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_Y: float, delta_Z: float
    ) -> ArrayLike:
        raise NotImplementedError

    def _calculate_incart_outcart(
        self,
        X: ArrayLike,
        Y: ArrayLike,
        Z: ArrayLike,
        vector: bool = False,
        unitvector: bool = False,
        magnitude: bool = False,
    ) -> List[FloatArray]:
        toreturn = []
        B_X = self.B_X(X, Y, Z)
        B_Y = self.B_Y(X, Y, Z)
        B_Z = self.B_Z(X, Y, Z)
        B_vec = np.array([B_X, B_Y, B_Z])  # (3,N)

        if vector:
            toreturn.append(B_vec.T)
        if unitvector or magnitude:
            B_mag = np.sqrt(B_X**2 + B_Y**2 + B_Z**2)
            if unitvector:
                b_hat = B_vec / B_mag
                toreturn.append(b_hat.T)
            if magnitude:
                toreturn.append(B_mag)

        return toreturn

    def all(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> List[FloatArray]:
        r"""Returns, in this order: :math:`\mathbf{B}`, :math:`\mathbf{B}/|B|`, and :math:`|B|`,
        which are the the vector, unit vector, and magnitude of the magnetic field respectively

        Input: cartesian (X, Y, Z)

        Output: cartesian
        """
        return self._calculate_incart_outcart(
            X, Y, Z, vector=True, unitvector=True, magnitude=True
        )

    def vector(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> FloatArray:
        r"""Returns :math:`\mathbf{B}`, the vector of the magnetic field

        Input: cartesian (X, Y, Z)

        Output: cartesian (B_X, B_Y, B_Z)
        """
        return self._calculate_incart_outcart(
            X, Y, Z, vector=True, unitvector=False, magnitude=False
        )[0]

    def unitvector(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> FloatArray:
        r"""Returns :math:`\mathbf{B}/|B|`, the unit vector of the magnetic field

        Input: cartesian (X, Y, Z)

        Output: cartesian (b_X, b_Y, b_Z)
        """
        return self._calculate_incart_outcart(
            X, Y, Z, vector=False, unitvector=True, magnitude=False
        )[0]

    def magnitude(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> ArrayLike:
        r"""Returns :math:`|B|`, the magnitude of the magnetic field

        Input: cartesian (X, Y, Z)
        """
        return self._calculate_incart_outcart(
            X, Y, Z, vector=False, unitvector=False, magnitude=True
        )[0]

    # For abstraction purposes
    def unitvector_cartesian(
        self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike
    ) -> FloatArray:
        return self.unitvector(X, Y, Z)

    def polflux_cartesian(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> ArrayLike:
        return self.polflux(X, Y, Z)


class InterpolatedField_Cartesian(MagneticField_Cartesian):
    def __init__(
        self,
        X_coord: FloatArray,
        Y_coord: FloatArray,
        Z_coord: FloatArray,
        B_X: FloatArray,
        B_Y: FloatArray,
        B_Z: FloatArray,
        psi: FloatArray,
        interp_order: str = "quintic",
    ):

        # Defining class attributes for the interpolated grids
        self.X_coord = X_coord
        self.Y_coord = Y_coord
        self.Z_coord = Z_coord
        self.B_X_grid = B_X
        self.B_Y_grid = B_Y
        self.B_Z_grid = B_Z
        self.psi_grid = psi
        self.interp_order = interp_order

        (self._interp_B_X, self._spline_B_X), duration_B_X_interpolation = (
            _make_cuboid_spline(X_coord, Y_coord, Z_coord, B_X, interp_order)
        )
        log.debug(f"Interpolating 3D B_X profile took {duration_B_X_interpolation} s")

        (self._interp_B_Y, self._spline_B_Y), duration_B_Y_interpolation = (
            _make_cuboid_spline(X_coord, Y_coord, Z_coord, B_Y, interp_order)
        )
        log.debug(f"Interpolating 3D B_Y profile took {duration_B_Y_interpolation} s")

        (self._interp_B_Z, self._spline_B_Z), duration_B_Z_interpolation = (
            _make_cuboid_spline(X_coord, Y_coord, Z_coord, B_Z, interp_order)
        )
        log.debug(f"Interpolating 3D B_Z profile took {duration_B_Z_interpolation} s")

        (
            (self._interp_polflux, self._spline_polflux),
            duration_polflux_interpolation,
        ) = _make_cuboid_spline(X_coord, Y_coord, Z_coord, psi, interp_order)
        log.debug(
            f"Interpolating 3D poloidal flux profile took {duration_polflux_interpolation} s"
        )

        self.grid_coords = self._spline_B_X.grid

    # Defining the class attributes for B_X, B_Y, B_Z, and polflux
    def B_X(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> ArrayLike:
        return self._interp_B_X(X, Y, Z)

    def B_Y(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> ArrayLike:
        return self._interp_B_Y(X, Y, Z)

    def B_Z(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> ArrayLike:
        return self._interp_B_Z(X, Y, Z)

    def polflux(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> ArrayLike:
        return self._interp_polflux(X, Y, Z)

    # TODO -- 30 September 2026
    # Defining class attributes for first order derivatives of B field -- for future work?
    # Need to compare with dbhats in analysis.py
    # These are from scotty 3D
    # def d_B_X_dX(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> ArrayLike:
    #     return self._spline_B_X((X,Y,Z), nu=[1,0,0])
    # def d_B_Y_dX(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> ArrayLike:
    #     return self._spline_B_Y((X,Y,Z), nu=[1,0,0])
    # def d_B_Z_dX(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> ArrayLike:
    #     return self._spline_B_Z((X,Y,Z), nu=[1,0,0])
    # def d_B_X_dY(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> ArrayLike:
    #     return self._spline_B_X((X,Y,Z), nu=[0,1,0])
    # def d_B_Y_dY(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> ArrayLike:
    #     return self._spline_B_Y((X,Y,Z), nu=[0,1,0])
    # def d_B_Z_dY(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> ArrayLike:
    #     return self._spline_B_Z((X,Y,Z), nu=[0,1,0])
    # def d_B_X_dZ(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> ArrayLike:
    #     return self._spline_B_X((X,Y,Z), nu=[0,0,1])
    # def d_B_Y_dZ(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> ArrayLike:
    #     return self._spline_B_Y((X,Y,Z), nu=[0,0,1])
    # def d_B_Z_dZ(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> ArrayLike:
    #     return self._spline_B_Z((X,Y,Z), nu=[0,0,1])
    # TODO NEED TO RENAME: this isnt actually d(bhat)/dX, but d(Bvec)/dX
    # Need to use chain rule to get d(bhat)/dX, so might actually be less computationally expensive to use finite differences
    # def d_bhat_dX(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> ArrayLike:
    #     return np.array( [self.d_B_X_dX(X,Y,Z), self.d_B_Y_dX(X,Y,Z), self.d_B_Z_dX(X,Y,Z)] ).T

    # Defining the class attributes for the first- and second-order derivatives of polflux
    def d_polflux_dX(
        self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_X: float
    ) -> ArrayLike:
        return derivative(self.polflux, ("X"), {"X": X, "Y": Y, "Z": Z}, {"X": delta_X})

    def d_polflux_dY(
        self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_Y: float
    ) -> ArrayLike:
        return derivative(self.polflux, ("Y"), {"X": X, "Y": Y, "Z": Z}, {"Y": delta_Y})

    def d_polflux_dZ(
        self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_Z: float
    ) -> ArrayLike:
        return derivative(self.polflux, ("Z"), {"X": X, "Y": Y, "Z": Z}, {"Z": delta_Z})

    def d2_polflux_dX2(
        self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_X: float
    ) -> ArrayLike:
        return derivative(
            self.polflux, ("X", "X"), {"X": X, "Y": Y, "Z": Z}, {"X": delta_X}
        )

    def d2_polflux_dY2(
        self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_Y: float
    ) -> ArrayLike:
        return derivative(
            self.polflux, ("Y", "Y"), {"X": X, "Y": Y, "Z": Z}, {"Y": delta_Y}
        )

    def d2_polflux_dZ2(
        self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_Z: float
    ) -> ArrayLike:
        return derivative(
            self.polflux, ("Z", "Z"), {"X": X, "Y": Y, "Z": Z}, {"Z": delta_Z}
        )

    def d2_polflux_dXdY(
        self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_X: float, delta_Y: float
    ) -> ArrayLike:
        return derivative(
            self.polflux,
            ("X", "Y"),
            {"X": X, "Y": Y, "Z": Z},
            {"X": delta_X, "Y": delta_Y},
        )

    def d2_polflux_dXdZ(
        self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_X: float, delta_Z: float
    ) -> ArrayLike:
        return derivative(
            self.polflux,
            ("X", "Z"),
            {"X": X, "Y": Y, "Z": Z},
            {"X": delta_X, "Z": delta_Z},
        )

    def d2_polflux_dYdZ(
        self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike, delta_Y: float, delta_Z: float
    ) -> ArrayLike:
        return derivative(
            self.polflux,
            ("Y", "Z"),
            {"X": X, "Y": Y, "Z": Z},
            {"Y": delta_Y, "Z": delta_Z},
        )

    # TODO -- 30 September 2026
    # Defining class attributes for poloidal flux derivatives
    # These are from scotty 3D
    # def d_polflux_dX_TEST(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> ArrayLike:
    #     return self._spline_polflux((X,Y,Z), nu=[1,0,0])
    # def d_polflux_dY_TEST(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> ArrayLike:
    #     return self._spline_polflux((X,Y,Z), nu=[0,1,0])
    # def d_polflux_dZ_TEST(self, X: ArrayLike, Y: ArrayLike, Z: ArrayLike) -> ArrayLike:
    #     return self._spline_polflux((X,Y,Z), nu=[0,0,1])


##################################################
#
# CREATING MAGNETIC FIELD
#
##################################################


#: `find_B_method` names removed in 4.0.0 (lower case), with ``shot`` and
#: ``equil_time``, and the field constructors that replace them: the file and
#: the time are their arguments
REMOVED_FIND_B_METHODS = {
    "test": "`InterpolatedField_Cylindrical.from_npz(path, equil_time)`",
    "test_notime": "`InterpolatedField_Cylindrical.from_npz(path)`",
    "uda_saved": (
        "`EFITField.from_MAST_saved(path, equil_time)` for a MAST shot, or "
        "`EFITField.from_MAST_U_saved(path, equil_time)` for a MAST-U shot "
        "(with no `equil_time` for a file with one time)"
    ),
    "efitpp": "`EFITField.from_EFITpp(path, equil_time)`",
}

#: `find_B_method` names that read the file ``magnetic_data_path``
FILE_FIND_B_METHODS = ("torbeam", "omfit")


def removed_find_B_method_error(find_B_method: str) -> ValueError:
    """The error for a `find_B_method` in `REMOVED_FIND_B_METHODS`"""
    replacement = REMOVED_FIND_B_METHODS[find_B_method.lower()]
    return ValueError(
        f"`find_B_method` = '{find_B_method}' was removed in Scotty 4.0.0, with "
        f"`shot` and `equil_time`. Make the field with {replacement} (from "
        "`scotty.geometry`), and give it to `find_B_method`"
    )


def create_magnetic_geometry(
    geometry: Literal["cylindrical", "cartesian"],
    find_B_method: Union[str, MagneticField_Cylindrical, MagneticField_Cartesian],
    interp_order_str: str,
    interp_order_int: int,
    interp_smoothing: float,
    magnetic_data_path: Optional[pathlib.Path] = None,
    **kwargs: Optional[dict],
) -> Union[MagneticField_Cylindrical, MagneticField_Cartesian]:
    """The magnetic field: ``find_B_method`` itself if it is a field object,
    else the field it names. ``"torbeam"`` and ``"omfit"`` read the file
    ``magnetic_data_path``; ``"analytical"`` takes ``B_T_axis``, ``R_axis``,
    ``minor_radius_a`` and ``B_p_a`` from ``kwargs``"""

    log.debug(f"Reading and creating field profile")

    def missing_arg(argument: str) -> str:
        return f"Missing '{argument}' for find_B_method='{find_B_method}'"

    # If the user passes an interpolated field, then just use that
    if (
        isinstance(
            find_B_method, (MagneticField_Cylindrical, InterpolatedField_Cylindrical)
        )
        and geometry == "cylindrical"
    ) or (
        isinstance(
            find_B_method, (MagneticField_Cartesian, InterpolatedField_Cartesian)
        )
        and geometry == "cartesian"
    ):
        log.debug(
            f"Using existing field profile of type `{type(find_B_method)}` passed from `find_B_method`"
        )
        return find_B_method

    # If the interpolated field doesnt match the user's geometry, raise an error
    elif (
        isinstance(
            find_B_method, (MagneticField_Cylindrical, InterpolatedField_Cylindrical)
        )
        and geometry == "cartesian"
    ) or (
        isinstance(
            find_B_method, (MagneticField_Cartesian, InterpolatedField_Cartesian)
        )
        and geometry == "cylindrical"
    ):
        raise RuntimeError(
            f"Using existing field profile of type `{type(find_B_method)}` passed from `find_B_method` is incompatible with `geometry` = {geometry}"
        )

    # TODO to be refactored
    # elif isinstance(find_B_method, (MagneticField_Cylindrical, InterpolatedField_Cylindrical)) and geometry == "cartesian":
    # else:
    #     log.info(f"Creating a 3-D field profile from the given 2-D profile")
    #     min_X, max_X = np.min(find_B_method.R_coord), np.max(find_B_method.R_coord)
    #     min_Z, max_Z = np.min(find_B_method.Z_coord), np.max(find_B_method.Z_coord)
    #     X_coords = (
    #         find_B_method.R_coord
    #     )  # np.linspace(min_X, max_X, round((max_X-min_X)/0.01+1))
    #     Y_coords = np.linspace(-1, 1, 101)  # np.linspace(-0.25, 0.1, 36)
    #     Z_coords = (
    #         find_B_method.Z_coord
    #     )  # np.linspace(min_Z, max_Z, round((max_Z-min_Z)/0.01+1))
    #     XX, YY, ZZ = np.meshgrid(X_coords, Y_coords, Z_coords, indexing="ij")
    #     RR = np.sqrt(XX**2 + YY**2)
    #     B_R = find_B_method.B_R(RR, ZZ)
    #     B_T = find_B_method.B_T(RR, ZZ)
    #     B_X = (B_R * XX - B_T * YY) / RR
    #     B_Y = (B_R * YY + B_T * XX) / RR
    #     B_Z = find_B_method.B_Z(RR, ZZ)
    #     polflux = find_B_method.poloidal_flux(RR, ZZ)
    #     field, duration_field_interpolation = timer(InterpolatedField_Cartesian)(
    #         X_coords, Y_coords, Z_coords, B_X, B_Y, B_Z, polflux, interp_order_str
    #     )

    #     log.info(
    #         f"Converting the B and poloidal flux profiles took {duration_field_interpolation} s"
    #     )
    #     return field

    # If a string, then interpolate as per normal
    elif isinstance(find_B_method, str):

        find_B_method = find_B_method.lower()

        if find_B_method == "analytical":
            B_T_axis = kwargs.get("B_T_axis")
            R_axis = kwargs.get("R_axis")
            minor_radius_a = kwargs.get("minor_radius_a")
            B_p_a = kwargs.get("B_p_a")
            if B_T_axis is None:
                raise ValueError(missing_arg("B_T_axis"))
            if R_axis is None:
                raise ValueError(missing_arg("R_axis"))
            if minor_radius_a is None:
                raise ValueError(missing_arg("minor_radius_a"))
            if B_p_a is None:
                raise ValueError(missing_arg("B_p_a"))
            # "analytic" (exact) or "numerical" (finite difference)
            # poloidal flux derivatives
            derivative_method = kwargs.get("derivative_method", "analytic")
            log.debug("Using analytical circular flux surface geometry")
            field = CircularFluxSurfaceField(
                B_T_axis,
                R_axis,
                minor_radius_a,
                B_p_a,
                derivative_method=derivative_method,
            )

        elif find_B_method in FILE_FIND_B_METHODS:
            if magnetic_data_path is None:
                raise ValueError(missing_arg("magnetic_data_path"))
            (
                q0,
                q1,
                q2,
                B0,
                B1,
                B2,
                psi,
            ) = get_profile_data(
                find_B_method=find_B_method.lower(),
                geometry=geometry,
                magnetic_data_path=pathlib.Path(magnetic_data_path),
            )

            if geometry == "cylindrical":
                (
                    field,
                    duration_field_interpolation,
                ) = timer(InterpolatedField_Cylindrical)(
                    R_coord=q0,
                    Z_coord=q2,
                    B_R=B0,
                    B_T=B1,
                    B_Z=B2,
                    psi=psi,
                    interp_order=interp_order_int,
                    interp_smoothing=interp_smoothing,
                )
            else:  # geometry == "cartesian":
                (
                    field,
                    duration_field_interpolation,
                ) = timer(InterpolatedField_Cartesian)(
                    X_coord=q0,
                    Y_coord=q1,
                    Z_coord=q2,
                    B_X=B0,
                    B_Y=B1,
                    B_Z=B2,
                    psi=psi,
                    interp_order=interp_order_str,
                )

            log.info(
                f"Interpolating B and poloidal flux profiles took {duration_field_interpolation} s"
            )

        elif find_B_method in REMOVED_FIND_B_METHODS:
            raise removed_find_B_method_error(find_B_method)

        else:
            raise ValueError(f"`find_B_method` = '{find_B_method}' is not supported")

        return field

    else:
        raise ValueError(f"`find_B_method` = '{find_B_method}' is not supported")
