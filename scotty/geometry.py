from abc import ABC, abstractmethod
import json
from multiprocessing.managers import ValueProxy
import numpy as np
import pathlib
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
from scotty.derivatives import derivative
from scotty.fun_general import find_q_labframe_cart_to_cyl
from scotty.logger import logging, timer
from scotty.torbeam import Torbeam
from scotty.typing import ArrayLike, FloatArray
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
    input_filename_suffix: str,
) -> Tuple[
    FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray
]:

    ##############################################
    # OMFIT
    ##############################################
    if find_B_method == "omfit":
        log.info(f"Using OMFIT JSON Torbeam file for B and poloidal flux")

        def unflatten(R_coord, Z_coord, arr):
            """Convert from column-major (TORBEAM, Fortran) to row-major order (Scotty, Python)"""
            return np.asarray(arr).reshape(len(Z_coord), len(R_coord)).T

        topfile_path = magnetic_data_path / f"topfile{input_filename_suffix}.json"
        if geometry == "cylindrical":
            with open(topfile_path) as f:
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
    # TODO: this should be `elif`. As an `if`, find_B_method = "omfit" loads
    # the data above, then falls through to the `else` below and raises
    # "not supported"; see "Known issues in v4" in README
    if find_B_method == "torbeam":
        log.info(f"Using TORBEAM input files for B and poloidal flux")
        topfile_path = magnetic_data_path / f"topfile{input_filename_suffix}"
        if geometry == "cylindrical":
            torbeam = Torbeam.from_file(topfile_path)
            q0 = torbeam.R_grid
            q1 = np.array([0])
            q2 = torbeam.Z_grid
            B0 = torbeam.B_R
            B1 = torbeam.B_T
            B2 = torbeam.B_Z
            psi = torbeam.psi
        else:  # geometry == "cartesian":
            raise ValueError(
                f"`find_B_method` = 'omfit' only works for `geometry` = 'cylindrical' for now"
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


@timer
def _make_cuboid_spline(
    X_coord, Y_coord, Z_coord, data_array, interp_order_str: str
) -> Tuple[
    Callable[[ArrayLike, ArrayLike, ArrayLike], FloatArray], RegularGridInterpolator
]:
    spline = RegularGridInterpolator(
        points=(X_coord, Y_coord, Z_coord),
        values=data_array,
        method=interp_order_str,
        bounds_error=False,
    )
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


class InterpolatedField_Cylindrical(MagneticField_Cylindrical):
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


class CircularFluxSurfaceField(MagneticField_Cylindrical):
    r"""Base class for analytical geometries with circular flux surfaces,
    where the poloidal flux is :math:`\psi = \rho / a`, with
    :math:`\rho = \sqrt{(R - R_{axis})^2 + Z^2}` and :math:`a` the minor radius.

    Child classes must set ``R_axis`` and ``minor_radius_a``, and implement
    ``B_R``, ``B_T`` and ``B_Z``.

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


class ConstantCurrentDensityField(CircularFluxSurfaceField):
    """Circular cross-section magnetic geometry with constant current density

    TODO
    ----
    Poloidal flux needs to be made consistent with B_{R,Z}

    Parameters
    ----------
    B_T_axis : float
        Toroidal magnetic field at the magnetic axis (Tesla)
    R_axis : float
        Major radius of the magnetic axis (metres)
    minor_radius_a : float
        Minor radius of the last closed flux surface (metres)
    B_p_a : float
        Poloidal magnetic field at ``minor_radius_a`` (Tesla)
    R_points : int
    Z_points : int
        Number of points for sample ``(R, Z)`` grid
    grid_buffer_factor:
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
        grid_buffer_factor: float = 1,
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
        return self.B_p_a * Z / self.rho(R, _, Z)

    def B_T(self, R: ArrayLike, _: ArrayLike, Z: ArrayLike) -> FloatArray:
        R, Z = np.asarray(R, dtype=np.float64), np.asarray(Z, dtype=np.float64)
        return self.B_T_axis * (self.R_axis / R)

    def B_Z(self, R: ArrayLike, _: ArrayLike, Z: ArrayLike) -> FloatArray:
        R, Z = np.asarray(R, dtype=np.float64), np.asarray(Z, dtype=np.float64)
        return -self.B_p_a * (R - self.R_axis) / self.rho(R, _, Z)


class CircularCrossSectionField(CircularFluxSurfaceField):
    """Simple circular cross-section magnetic geometry

    Parameters
    ----------
    B_T_axis : float
        Toroidal magnetic field at the magnetic axis (Tesla)
    R_axis : float
        Major radius of the magnetic axis (metres)
    minor_radius_a : float
        Minor radius of the last closed flux surface (metres)
    B_p_a : float
        Poloidal magnetic field at ``minor_radius_a`` (Tesla)
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
        return np.where(
            abs(Z) < 1e-12,
            0.0,
            self.B_p_a * Z / (R * self.minor_radius_a * self.rho(R, _, Z)),
        )

    def B_T(self, R: ArrayLike, _: ArrayLike, Z: ArrayLike) -> FloatArray:
        R, Z = np.asarray(R, dtype=np.float64), np.asarray(Z, dtype=np.float64)
        return self.B_T_axis * (self.R_axis / R)

    def B_Z(self, R: ArrayLike, _: ArrayLike, Z: ArrayLike) -> FloatArray:
        R, Z = np.asarray(R, dtype=np.float64), np.asarray(Z, dtype=np.float64)
        return np.where(
            abs(R - self.R_axis) < 1e-12,
            0.0,
            -self.B_p_a
            * (R - self.R_axis)
            / (R * self.minor_radius_a * self.rho(R, _, Z)),
        )


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


def create_magnetic_geometry(
    geometry: Literal["cylindrical", "cartesian"],
    find_B_method: Union[str, MagneticField_Cylindrical, MagneticField_Cartesian],
    interp_order_str: str,
    interp_order_int: int,
    interp_smoothing: int,
    magnetic_data_path: pathlib.Path,
    input_filename_suffix: str = "",
    shot: Optional[int] = None,
    equil_time: Optional[float] = None,
    **kwargs: Optional[dict],
) -> Union[MagneticField_Cylindrical, MagneticField_Cartesian]:

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

        if find_B_method in ["analytical", "unit-tests"]:
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
            if find_B_method == "analytical":
                log.debug("Using analytical constant current density geometry")
                field = ConstantCurrentDensityField(
                    B_T_axis,
                    R_axis,
                    minor_radius_a,
                    B_p_a,
                    derivative_method=derivative_method,
                )
            else:
                log.debug("Using analytical circular cross-section geometry")
                field = CircularCrossSectionField(
                    B_T_axis,
                    R_axis,
                    minor_radius_a,
                    B_p_a,
                    derivative_method=derivative_method,
                )

        elif find_B_method in ["omfit", "torbeam"]:
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
                magnetic_data_path=magnetic_data_path,
                input_filename_suffix=input_filename_suffix,
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

        # TODO: port the remaining v3 sources ("test", "test_notime",
        # "UDA_saved", "EFITpp", "curvy_slab") to v4
        else:
            raise ValueError(f"`find_B_method` = '{find_B_method}' is not supported")

        return field

    else:
        raise ValueError(f"`find_B_method` = '{find_B_method}' is not supported")
