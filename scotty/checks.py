# -*- coding: utf-8 -*-
# Copyright 2018 - 2023, Valerian Hall-Chen and the Scotty contributors
# SPDX-License-Identifier: GPL-3.0

import difflib
import logging
import warnings
from pathlib import Path
from scotty.compat import REMOVED_ARGUMENTS, V3_ONLY_ARGUMENTS
from scotty.fun_general import (
    find_K_labframe_cart_to_cyl,
    find_K_labframe_cyl_to_cart,
    find_K_magnitude,
    find_Psi_3D_labframe_cart_to_cyl,
    find_Psi_3D_labframe_cyl_to_cart,
    find_q_labframe_cart_to_cyl,
    find_q_labframe_cyl_to_cart,
)
from scotty.geometry import (
    FILE_FIND_B_METHODS,
    REMOVED_FIND_B_METHODS,
    MagneticField_Cylindrical,
    MagneticField_Cartesian,
    InterpolatedField_Cylindrical,
    InterpolatedField_Cartesian,
    removed_find_B_method_error,
)
from scotty.profile_fit import ProfileFitLike
from scotty.typing import ArrayLike, FloatArray, ComplexFloatArray
from typing import Literal, Optional, Sequence, Union, Tuple, cast, get_args
import numpy as np
import os

##################################################
#
# CHECKS
#
##################################################

VALID_GEOMETRIES = Literal["cylindrical", "cartesian"]
VALID_LAUNCH_MODE_FLAGS = Literal[1, -1, "O", "X"]
VALID_LAUNCH_FLAGS = Literal["vacuum_analytic", "vacuum_numerical", "plasma"]
VALID_PSI_BC_FLAGS = Optional[Literal["continuous", "discontinuous"]]
# What the ray does at a resonance layer (`first_harmonic_flag`,
# `second_harmonic_flag`, `upper_hybrid_flag`; see `Parameters._check_resonance_flag`)
VALID_RESONANCE_FLAGS = Literal["stop", "raise", "pass"]
VALID_FIELDS = Union[
    MagneticField_Cylindrical,
    MagneticField_Cartesian,
    InterpolatedField_Cylindrical,
    InterpolatedField_Cartesian,
]
VALID_SOLVER_STATUS = Literal[-1, 0, 1]

##################################################
#
# CLASS PARAMETERS
#
##################################################

log = logging.getLogger(__name__)

# Keyword arguments that `beam_me_up` takes through `**kwargs`
EXTRA_ARGUMENTS = (
    "B_T_axis",
    "B_p_a",
    "R_axis",
    "minor_radius_a",
    "plasmaLaunch_K_cartesian",
    "plasmaLaunch_Psi_3D_lab_cartesian",
    "create_magnetic_geometry",
)

#: Arguments removed in 4.0.0, and what replaces them. They raise an error
#: rather than the warning for an unknown keyword, since ignoring them would
#: change the run
REPLACED_ARGUMENTS = {
    "interp_order": "`interp_order_magnetic`, `interp_order_ne` and `interp_order_Te`",
    "interp_smoothing": (
        "`interp_smoothing_magnetic`, `interp_smoothing_ne` and `interp_smoothing_Te`"
    ),
    "input_filename_suffix": (
        "the paths of the files themselves, in `magnetic_data_path`, "
        "`ne_data_path` and `Te_data_path`"
    ),
    "shot": (
        "a field object in `find_B_method`, made from the file with a "
        "constructor in `scotty.geometry`, for example "
        "`EFITField.from_MAST_saved(path, equil_time)`"
    ),
    "equil_time": (
        "a field object in `find_B_method`, made from the file with a "
        "constructor in `scotty.geometry` that takes the time, for example "
        "`EFITField.from_MAST_saved(path, equil_time)`"
    ),
}

#: An example file name for each data path, for the error when a folder is given
_EXAMPLE_FILE_NAMES = {
    "magnetic_data_path": "topfile",
    "ne_data_path": "ne.dat",
    "Te_data_path": "Te.dat",
}


def _reads_file(fit_method) -> bool:
    """Whether a profile method reads a file (``"smoothing-spline-file"``)"""
    return isinstance(fit_method, str) and fit_method == "smoothing-spline-file"


LaunchBeamArgument = Union[float, Tuple[float, float], Tuple[float, float, float]]
"""`launch_beam_width` and `launch_beam_curvature`: a number for a circular beam.
A 2-tuple and a 3-tuple are reserved for elliptical and astigmatic beams, and
are not implemented yet"""


def _circular_launch_beam(name: str, value: LaunchBeamArgument) -> float:
    """`value` as a number, for a circular beam. A 2-tuple or 3-tuple (an
    elliptical or astigmatic beam) raises `NotImplementedError`: these forms
    are reserved until the launch beam frame and their meaning are decided"""
    array = np.asarray(value) if value is not None else np.asarray(np.nan, dtype=object)
    if array.ndim == 0 and array.dtype.kind in "iuf":
        if not np.isfinite(array):
            raise ValueError(f"`{name}` must be finite, but got {value!r}")
        return float(array)
    if array.ndim == 1 and len(array) in (2, 3) and array.dtype.kind in "iuf":
        raise NotImplementedError(
            f"`{name}` = {value}: a {len(array)}-tuple (an elliptical or "
            "astigmatic beam) is reserved, but not implemented yet. Give a "
            "number, for a circular beam"
        )
    raise ValueError(f"`{name}` must be a number (a circular beam), but got {value!r}")


def _known_argument_names() -> list:
    """The names of `beam_me_up`'s arguments, to suggest the intended name of
    a misspelt keyword argument"""
    import inspect

    # Imported here, since `scotty.beam_me_up` imports this module
    from scotty.beam_me_up import beam_me_up

    parameters = inspect.signature(beam_me_up).parameters
    return [
        name
        for name, parameter in parameters.items()
        if parameter.kind != inspect.Parameter.VAR_KEYWORD
    ] + list(EXTRA_ARGUMENTS)


class Parameters:
    r"""The inputs of `scotty.beam_me_up.beam_me_up`, checked, and the
    quantities that the run works out from them (positions, wavevectors and
    Psi at the launch, the entry and the initial point, and the solver
    output). `beam_me_up` makes one, and `scotty.data_io` saves it in the
    output. Its arguments are those of `beam_me_up` (``launch_position`` is
    ``q_launch``, and ``mode_flag`` is ``mode_flag_launch``); an argument that
    is not valid raises a `ValueError` here, before any computation."""

    # Declared here to stop type checker from complaining
    geometry: VALID_GEOMETRIES
    cartesian_flag: bool
    launch_flag: VALID_LAUNCH_FLAGS
    Psi_BC_flag: VALID_PSI_BC_FLAGS
    mode_flag_launch: VALID_LAUNCH_MODE_FLAGS
    mode_flag_initial: Literal[1, -1]
    mode_index: int
    e_hat_initial: ComplexFloatArray

    # miscellaneous
    solver_status: VALID_SOLVER_STATUS
    solver_nfev: int
    solver_duration: float
    tau_output: FloatArray

    # position vectors
    q_initial: FloatArray
    q_output: FloatArray
    distance_from_launch_to_entry: Optional[float]

    # wavevectors
    K_launch: FloatArray
    K_initial: FloatArray
    K_output: FloatArray
    K_output_magnitude: FloatArray
    K_hat_output_cartesian: FloatArray

    # beam matrices
    Psi_3D_launch_labframe: Optional[ComplexFloatArray]
    Psi_3D_entry_labframe: Optional[ComplexFloatArray]
    Psi_3D_initial_labframe: Optional[ComplexFloatArray]
    Psi_3D_output_labframe: Optional[ComplexFloatArray]

    def __init__(
        self,
        # Main parameters
        geometry: VALID_GEOMETRIES,
        poloidal_launch_angle_Torbeam: float,
        toroidal_launch_angle_Torbeam: float,
        launch_freq_GHz: float,
        launch_beam_width: LaunchBeamArgument,
        launch_beam_curvature: LaunchBeamArgument,
        q_launch: FloatArray,
        mode_flag_launch: VALID_LAUNCH_MODE_FLAGS,
        # Solver settings and finite-difference parameters
        launch_flag: VALID_LAUNCH_FLAGS,
        Psi_BC_flag: VALID_PSI_BC_FLAGS,
        relativistic_flag: bool,
        first_harmonic_flag: VALID_RESONANCE_FLAGS,
        second_harmonic_flag: VALID_RESONANCE_FLAGS,
        upper_hybrid_flag: VALID_RESONANCE_FLAGS,
        auto_delta_sign: bool,
        deltas_cylindrical: FloatArray,
        deltas_cartesian: FloatArray,
        len_tau: int,
        rtol: float,
        atol: float,
        poloidal_flux_enter: Optional[float],
        poloidal_flux_zero_density: Optional[float],
        poloidal_flux_zero_temperature: Optional[float],
        # Data input and output arguments
        magnetic_data_path: Optional[Union[str, Path]],
        ne_data_path: Optional[Union[str, Path]],
        Te_data_path: Optional[Union[str, Path]],
        output_path: Union[str, Path],
        output_filename_suffix: str,
        # Sources of the field and the profiles, and interpolation settings
        find_B_method: Union[str, MagneticField_Cylindrical, MagneticField_Cartesian],
        density_fit_method: Union[str, ProfileFitLike],
        density_fit_parameters: Optional[Sequence],
        temperature_fit_method: Optional[Union[str, ProfileFitLike]],
        temperature_fit_parameters: Optional[Sequence],
        interp_order_magnetic: Union[str, int],
        interp_smoothing_magnetic: float,
        interp_order_ne: Union[str, int],
        interp_smoothing_ne: float,
        interp_order_Te: Union[str, int],
        interp_smoothing_Te: float,
        # Plotting flags
        figure_flag: bool,
        # Analysis flags
        analysis_flag: bool,
        dbs_analysis_flag: bool,
        # Additional flags
        ray_tracing_flag: bool,
        # return_dt_field: bool,
        # Extra kwargs for parsing
        **kwargs,
    ):

        from scotty.fun_general import (
            freq_GHz_to_angular_frequency,
            angular_frequency_to_wavenumber,
        )

        ##################################################
        #
        # Main parameters
        #
        ##################################################

        self.geometry = self._check_geometry(geometry=geometry)
        self.cartesian_flag = self.geometry == "cartesian"

        # The launch angles stay in the TORBEAM convention, measured from the
        # local -R at the antenna: `launch_direction_cartesian` turns them into
        # the launch direction. (The commented lines below are an older
        # conversion, no longer used)
        self.poloidal_launch_angle_deg_Torbeam = poloidal_launch_angle_Torbeam
        self.poloidal_launch_angle_rad_Torbeam = np.deg2rad(
            self.poloidal_launch_angle_deg_Torbeam
        )
        self.toroidal_launch_angle_deg_Torbeam = toroidal_launch_angle_Torbeam
        self.toroidal_launch_angle_rad_Torbeam = np.deg2rad(
            self.toroidal_launch_angle_deg_Torbeam
        )
        # self.poloidal_launch_angle_rad = -self.poloidal_launch_angle_rad_Torbeam
        # self.poloidal_launch_angle_deg = np.rad2deg(self.poloidal_launch_angle_rad)
        # self.toroidal_launch_angle_rad = self.toroidal_launch_angle_rad_Torbeam + np.pi
        # self.toroidal_launch_angle_deg = np.rad2deg(self.toroidal_launch_angle_rad)

        self.launch_frequency_GHz = self._check_positive(
            "launch_freq_GHz", launch_freq_GHz
        )
        self.launch_angular_frequency = freq_GHz_to_angular_frequency(
            self.launch_frequency_GHz
        )
        self.launch_wavenumber = angular_frequency_to_wavenumber(
            self.launch_angular_frequency
        )
        self.launch_beam_width = self._check_positive(
            "launch_beam_width",
            _circular_launch_beam("launch_beam_width", launch_beam_width),
        )
        self.launch_beam_curvature = _circular_launch_beam(
            "launch_beam_curvature", launch_beam_curvature
        )
        self.q_launch = q_launch
        self.mode_flag_launch = self._check_mode_flag(mode_flag_launch)

        ##################################################
        #
        # Solver settings and finite-difference parameters
        #
        ##################################################

        self.ray_tracing_flag = ray_tracing_flag
        self.launch_flag = self._check_launch_flag(launch_flag)
        self.Psi_BC_flag = self._check_Psi_BC_flag(Psi_BC_flag, self.launch_flag)
        self.relativistic_flag = relativistic_flag
        self._check_replaced_kwargs(extras=kwargs)
        self._check_methods_given(
            find_B_method, density_fit_method, temperature_fit_method
        )
        self.first_harmonic_flag = self._check_resonance_flag(
            "first_harmonic_flag", first_harmonic_flag
        )
        self.second_harmonic_flag = self._check_resonance_flag(
            "second_harmonic_flag", second_harmonic_flag
        )
        self.upper_hybrid_flag = self._check_resonance_flag(
            "upper_hybrid_flag", upper_hybrid_flag
        )
        self.auto_delta_sign = auto_delta_sign
        self.len_tau = int(self._check_positive("len_tau", len_tau))
        self.rtol = rtol
        self.atol = atol
        (
            self.poloidal_flux_enter,
            self.poloidal_flux_zero_density,
            self.poloidal_flux_zero_temperature,
        ) = self._check_poloidal_flux_arguments(
            poloidal_flux_enter,
            poloidal_flux_zero_density,
            poloidal_flux_zero_temperature,
            relativistic_flag,
            density_fit_method,
            temperature_fit_method,
        )

        ##################################################
        #
        # Data input, output, plotting arguments
        #
        ##################################################

        self.find_B_method = (
            find_B_method
            if isinstance(find_B_method, str)
            else str(type(find_B_method))
        )
        # Each data path is the path of a file, checked only if the run reads it
        self.magnetic_data_path = self._check_data_file(
            "magnetic_data_path",
            magnetic_data_path,
            used=isinstance(find_B_method, str)
            and find_B_method.lower() in FILE_FIND_B_METHODS,
            what="magnetic field file (a TORBEAM topfile, or its OMFIT JSON form)",
            unused="`find_B_method` does not read a file",
        )
        self.interp_order_magnetic_str, self.interp_order_magnetic = (
            self._check_interp_order("interp_order_magnetic", interp_order_magnetic)
        )
        self.interp_smoothing_magnetic = self._check_smoothing(
            "interp_smoothing_magnetic", interp_smoothing_magnetic
        )

        self.density_fit_method = density_fit_method
        self.density_fit_parameters = self._check_file_fit_parameters(
            "density", density_fit_method, density_fit_parameters
        )
        self.ne_data_path = self._check_data_file(
            "ne_data_path",
            ne_data_path,
            used=_reads_file(density_fit_method),
            what="electron density file",
            unused="`density_fit_method` is not 'smoothing-spline-file'",
        )
        _, self.interp_order_ne = self._check_interp_order(
            "interp_order_ne", interp_order_ne
        )
        self.interp_smoothing_ne = self._check_smoothing(
            "interp_smoothing_ne", interp_smoothing_ne
        )

        # The temperature is used only by relativistic runs
        self.temperature_fit_method = temperature_fit_method
        self.temperature_fit_parameters = (
            self._check_file_fit_parameters(
                "temperature", temperature_fit_method, temperature_fit_parameters
            )
            if relativistic_flag
            else temperature_fit_parameters
        )
        self.Te_data_path = self._check_data_file(
            "Te_data_path",
            Te_data_path,
            used=relativistic_flag and _reads_file(temperature_fit_method),
            what="electron temperature file",
            unused="the run is not relativistic, or `temperature_fit_method` is "
            "not 'smoothing-spline-file'",
        )
        _, self.interp_order_Te = self._check_interp_order(
            "interp_order_Te", interp_order_Te
        )
        self.interp_smoothing_Te = self._check_smoothing(
            "interp_smoothing_Te", interp_smoothing_Te
        )

        self.output_path = self._check_output_path(output_path)
        self.output_filename_suffix = output_filename_suffix

        self.figure_flag = figure_flag
        self.analysis_flag, self.dbs_analysis_flag = self._check_analysis_flags(
            analysis_flag, dbs_analysis_flag
        )

        self._unpack_kwargs_for_circular_flux_surfaces(extras=kwargs)
        self._unpack_kwargs_for_plasma_launch(extras=kwargs)
        self._unpack_unused_kwargs(extras=kwargs)

        ##################################################
        #
        # Geometry-specific parameters
        #
        ##################################################

        if geometry == "cylindrical":
            (
                self.delta_R,
                self.delta_Z,
                self.delta_K_R,
                self.delta_K_zeta,
                self.delta_K_Z,
            ) = deltas_cylindrical

            self.deltas = deltas_cylindrical

        elif geometry == "cartesian":
            (
                self.delta_X,
                self.delta_Y,
                self.delta_Z,
                self.delta_K_X,
                self.delta_K_Y,
                self.delta_K_Z,
            ) = deltas_cartesian

            self.deltas = deltas_cartesian

        else:
            raise ValueError(
                f"`geometry` must be one of 'cylindrical' or 'cartesian' but got {geometry}"
            )

    ##################################################
    #
    # PRIMARY INPUT CHECKS (for beam_me_up)
    # For the inputs given to beam_me_up
    #
    # TODO: check the data in the density and temperature files:
    #    - rho must be sorted
    #    - must polflux_zero_density be less than the last polflux
    #      coordinate in ne.dat?
    #
    ##################################################

    def _check_geometry(self, geometry: str) -> VALID_GEOMETRIES:
        geometry = geometry.lower()
        if geometry not in get_args(VALID_GEOMETRIES):
            raise ValueError(
                f"`geometry` must be one of {get_args(VALID_GEOMETRIES)}, but got {geometry}"
            )
        else:
            return cast(VALID_GEOMETRIES, geometry)

    def _check_positive(self, name: str, value: Union[int, float]) -> Union[int, float]:
        # `not value > 0` also catches NaN, for which `value <= 0` is False
        if not value > 0:
            raise ValueError(f"`{name}` must be positive, but got {value}")
        else:
            return value

    def _check_mode_flag(
        self, mode_flag_launch: Union[str, int]
    ) -> VALID_LAUNCH_MODE_FLAGS:
        if mode_flag_launch not in get_args(VALID_LAUNCH_MODE_FLAGS):
            raise ValueError(
                f"`mode_flag` must be one of {get_args(VALID_LAUNCH_MODE_FLAGS)}, but got {mode_flag_launch}"
            )
        else:
            return cast(VALID_LAUNCH_MODE_FLAGS, mode_flag_launch)

    def _check_poloidal_flux_arguments(
        self,
        poloidal_flux_enter: Optional[float],
        poloidal_flux_zero_density: Optional[float],
        poloidal_flux_zero_temperature: Optional[float],
        relativistic_flag: bool,
        density_fit_method=None,
        temperature_fit_method=None,
    ) -> Tuple[float, float, float]:
        # A profile object (e.g. `QuadraticFit`) has its own zero point, which is
        # what the run uses; the arguments must agree with it, and default to it
        def zero_point(name, given, fit_method, default):
            own = getattr(fit_method, "poloidal_flux_zero_profile", None)
            if given is None:
                return default if own is None else own
            if own is not None and not np.isclose(given, own, rtol=1e-12, atol=0):
                raise ValueError(
                    f"`{name}` = {given}, but the profile {fit_method!r} is zero "
                    f"from {own}, which is what the run would use. Leave `{name}` "
                    "unset (None) to use the profile's value, or make them agree"
                )
            return given

        for name, given in (
            ("poloidal_flux_zero_density", poloidal_flux_zero_density),
            ("poloidal_flux_zero_temperature", poloidal_flux_zero_temperature),
        ):
            if given is not None:
                warnings.warn(
                    f"`{name}` is deprecated, and will be removed after Scotty 4.x: "
                    "give the zero point to the profile object instead, as the "
                    "first argument of the fits in `scotty.profile_fit` (for "
                    "example `QuadraticFit(1.0, 4.0)` for `'quadratic'` with "
                    "parameters `[4.0]`, or `SmoothingSplineFit.from_dat_file(1.0, "
                    "'ne.dat')` for a density file). Leaving it unset with a named "
                    "fit or a file means a zero point of 1.0. With "
                    "`poloidal_flux_enter` unset, the plasma starts where the "
                    "density does",
                    FutureWarning,
                    stacklevel=4,
                )

        poloidal_flux_zero_density = zero_point(
            "poloidal_flux_zero_density",
            poloidal_flux_zero_density,
            density_fit_method,
            1.0,
        )
        # By default, the temperature goes to zero where the density does
        poloidal_flux_zero_temperature = zero_point(
            "poloidal_flux_zero_temperature",
            poloidal_flux_zero_temperature,
            temperature_fit_method,
            poloidal_flux_zero_density,
        )
        self._check_positive("poloidal_flux_zero_density", poloidal_flux_zero_density)
        self._check_positive(
            "poloidal_flux_zero_temperature", poloidal_flux_zero_temperature
        )
        if poloidal_flux_enter is None:
            # Set where the density starts, once the profile exists
            # (`set_poloidal_flux_enter`)
            return (
                poloidal_flux_enter,
                poloidal_flux_zero_density,
                poloidal_flux_zero_temperature,
            )
        self._check_positive("poloidal_flux_enter", poloidal_flux_enter)
        self._check_zero_points_against_entry(
            poloidal_flux_enter,
            poloidal_flux_zero_density,
            poloidal_flux_zero_temperature,
            relativistic_flag,
        )
        return (
            poloidal_flux_enter,
            poloidal_flux_zero_density,
            poloidal_flux_zero_temperature,
        )

    @staticmethod
    def _check_zero_points_against_entry(
        poloidal_flux_enter: float,
        poloidal_flux_zero_density: float,
        poloidal_flux_zero_temperature: float,
        relativistic_flag: bool,
    ) -> None:
        if poloidal_flux_zero_density < poloidal_flux_enter:
            raise ValueError(
                f"`poloidal_flux_zero_density` must be greater than or equal to `poloidal_flux_enter`, but got `poloidal_flux_zero_density` = {poloidal_flux_zero_density} and `poloidal_flux_enter` = {poloidal_flux_enter}"
            )
        # The temperature profile is used only by relativistic runs
        if relativistic_flag and poloidal_flux_zero_temperature < poloidal_flux_enter:
            raise ValueError(
                f"`poloidal_flux_zero_temperature` must be greater than or equal to `poloidal_flux_enter`, but got `poloidal_flux_zero_temperature` = {poloidal_flux_zero_temperature} and `poloidal_flux_enter` = {poloidal_flux_enter}"
                " (the temperature would be zero inside the plasma)"
            )
        if (
            poloidal_flux_zero_density != poloidal_flux_zero_temperature
            and relativistic_flag
        ):
            log.warning(
                f"`poloidal_flux_zero_density` ({poloidal_flux_zero_density}) and `poloidal_flux_zero_temperature` ({poloidal_flux_zero_temperature}) are set to different values. This may cause issues because `relativistic_flag` = {relativistic_flag}"
            )

    def set_poloidal_flux_enter(self, density_fit) -> None:
        """If `poloidal_flux_enter` was left unset: where the density starts,
        going in from its zero point (`find_density_start`), so that the
        boundary condition is applied there and the beam never crosses the
        start of the density inside the plasma"""
        if self.poloidal_flux_enter is not None:
            return
        from scotty.boundary_conditions import find_density_start

        # A spline from data that end inside the zero point holds its last
        # value beyond them, with a kink at the end of the data: start there
        # at the latest
        zero_point = self.poloidal_flux_zero_density
        data_end = getattr(density_fit, "poloidal_flux_data_end", None)
        if data_end is not None and data_end < zero_point:
            zero_point = data_end
        n_max = float(np.max(np.abs(density_fit(np.linspace(0.0, zero_point, 2001)))))
        start = (
            find_density_start(density_fit, zero_point, 1e-10 * n_max)
            if n_max > 0
            else 0.0
        )
        if start <= 0:
            raise ValueError(
                f"The density profile ({density_fit!r}) is zero everywhere inside "
                f"poloidal flux {zero_point}: give `poloidal_flux_enter`"
            )
        # A profile that falls to zero at its zero point crosses the threshold
        # just inside it (by 5e-6 for one that starts flat, 4(1 - psi^2)^2):
        # that is not a zero band, so enter at the zero point itself, as
        # `check_density_at_entry` does (the same `band`, 1e-4)
        if start >= zero_point * (1 - 1e-4):
            start = float(zero_point)
        self.poloidal_flux_enter = start
        log.info(f"`poloidal_flux_enter` = {start:.6g}: where the density starts")
        self._check_zero_points_against_entry(
            start,
            self.poloidal_flux_zero_density,
            self.poloidal_flux_zero_temperature,
            self.relativistic_flag,
        )

    def _check_data_file(
        self,
        name: str,
        path: Optional[Union[str, Path]],
        used: bool,
        what: str,
        unused: str,
    ) -> Optional[Path]:
        """``path`` as a `Path`: the file that the run reads (it must exist),
        or ``None``. A path that the run does not read gives a warning"""
        if path is None:
            if used:
                raise ValueError(f"`{name}` must be given: the path of the {what}")
            return None
        path = Path(path)
        if not used:
            log.warning(f"`{name}` = '{path}' is not used, because {unused}")
            return path
        if path.is_dir():
            raise ValueError(
                f"`{name}` = '{path}' is a folder. Since Scotty 4.0.0, `{name}` is "
                f"the path of the {what} itself, for example "
                f"'{path / _EXAMPLE_FILE_NAMES[name]}'"
            )
        if not path.is_file():
            raise ValueError(f"`{name}` = '{path}': there is no such file ({what})")
        return path

    @staticmethod
    def _check_output_path(path: Union[str, Path]) -> Path:
        """The folder for the output file, created if it does not exist"""
        path = Path(path)
        if not path.is_dir():
            log.info(f"`output_path` = {path} does not exist. Creating the folder")
            os.makedirs(path)
        return path

    @staticmethod
    def _check_file_fit_parameters(
        quantity: str, fit_method, parameters: Optional[Sequence]
    ) -> Optional[Sequence]:
        """``"smoothing-spline-file"`` takes its file and spline settings from
        their own arguments, so it takes no parameters"""
        if _reads_file(fit_method) and parameters is not None:
            short = "ne" if quantity == "density" else "Te"
            raise ValueError(
                f"`{quantity}_fit_method` = 'smoothing-spline-file' reads the file "
                f"`{short}_data_path`, with `interp_order_{short}` and "
                f"`interp_smoothing_{short}`, so `{quantity}_fit_parameters` must "
                f"not be given, but got {parameters!r}"
            )
        return parameters

    @staticmethod
    def _check_smoothing(name: str, smoothing) -> float:
        """A smoothing factor of a spline: a number of at least 0"""
        if (
            isinstance(smoothing, (bool, np.bool_))
            or not isinstance(smoothing, (int, float, np.integer, np.floating))
            or not np.isfinite(smoothing)
            or smoothing < 0
        ):
            raise ValueError(
                f"`{name}` must be a number of at least 0 (0 for no smoothing), "
                f"but got {smoothing!r}"
            )
        return float(smoothing)

    @staticmethod
    def _check_interp_order(name: str, order: Union[str, int]) -> Tuple[str, int]:
        """The order of a spline, as its name and its number: 1, 3 or 5, or
        "linear", "cubic" or "quintic"
        """
        _valid_interp_orders_int = {1: "linear", 3: "cubic", 5: "quintic"}
        _valid_interp_orders_str = dict(
            (v, k) for k, v in _valid_interp_orders_int.items()
        )

        # `True` would compare equal to 1
        if not isinstance(order, (bool, np.bool_)):
            if isinstance(order, str) and order in _valid_interp_orders_str:
                return order, _valid_interp_orders_str[order]
            if not isinstance(order, str) and order in _valid_interp_orders_int:
                return _valid_interp_orders_int[order], int(order)
        raise ValueError(
            f"`{name}` must be one of "
            f"{list(_valid_interp_orders_int) + list(_valid_interp_orders_str)}, "
            f"but got {order!r}"
        )

    def _check_launch_flag(self, launch_flag) -> VALID_LAUNCH_FLAGS:
        # Any `mode_flag` (checked in `_check_mode_flag`) works with any
        # launch: for "O" or "X", the mode is found from K where the solvers
        # start, for a plasma launch as for a vacuum one
        if not isinstance(launch_flag, str) or launch_flag not in get_args(
            VALID_LAUNCH_FLAGS
        ):
            raise ValueError(
                f"`launch_flag` must be one of {get_args(VALID_LAUNCH_FLAGS)}, but got {launch_flag!r}"
            )
        return cast(VALID_LAUNCH_FLAGS, launch_flag)

    def _check_resonance_flag(self, name: str, value) -> VALID_RESONANCE_FLAGS:
        # `isinstance` first, so that the old `resonance_flag`'s booleans and
        # integers are refused rather than compared with the strings
        if not isinstance(value, str) or value not in get_args(VALID_RESONANCE_FLAGS):
            raise ValueError(
                f"`{name}` must be one of {get_args(VALID_RESONANCE_FLAGS)}, but got "
                f"{value!r}"
            )
        # The cold-plasma Hamiltonian is singular at the first harmonic, and past
        # it the root that `mode_flag` selects swaps (beam-tracing/Scotty#169)
        if name == "first_harmonic_flag" and value == "pass":
            raise ValueError(
                "`first_harmonic_flag` can't be 'pass': the cold-plasma Hamiltonian is "
                "singular at the first harmonic of the electron cyclotron resonance "
                "(the fundamental), so the ray can't be traced past it. Use 'stop' to "
                "get the ray up to 1e-3 short of the layer, in omega_ce / omega, or "
                "'raise'"
            )
        # The X-mode's refractive index goes to infinity at the upper hybrid layer,
        # which a ray only approaches asymptotically
        if name == "upper_hybrid_flag" and value == "pass":
            raise ValueError(
                "`upper_hybrid_flag` can't be 'pass': the X-mode's refractive index "
                "goes to infinity at the upper hybrid resonance, so the ray can't be "
                "traced past it. Use 'stop' to get the ray up to 1e-2 short of the "
                "layer, in omega_UH / omega, or 'raise'"
            )
        return cast(VALID_RESONANCE_FLAGS, value)

    def _check_replaced_kwargs(self, **extras):
        # Raised rather than warned about as an unused keyword, since ignoring it
        # would change where the ray stops
        extras = extras["extras"]
        for name, replacement in REPLACED_ARGUMENTS.items():
            if name in extras:
                raise ValueError(
                    f"`{name}` (got {extras[name]!r}) was removed in Scotty 4.0.0: "
                    f"use {replacement}. To run a Scotty v3 script, pass its "
                    "arguments through `scotty.compat.from_v3_kwargs`, which "
                    "converts them"
                )
        if "resonance_flag" in extras:
            raise ValueError(
                f"`resonance_flag` (got {extras['resonance_flag']!r}) has been replaced "
                "by `first_harmonic_flag`, `second_harmonic_flag` and "
                "`upper_hybrid_flag`, each 'stop', 'raise' or 'pass'. To run a Scotty "
                "v3 script, pass its arguments through `scotty.compat.from_v3_kwargs`, "
                "which converts `resonance_flag`"
            )

    def _check_Psi_BC_flag(
        self, Psi_BC_flag, launch_flag: VALID_LAUNCH_FLAGS
    ) -> VALID_PSI_BC_FLAGS:
        # `isinstance` first, so that v3's booleans are refused rather than
        # compared with the strings
        if Psi_BC_flag is not None and (
            not isinstance(Psi_BC_flag, str)
            or Psi_BC_flag not in ["continuous", "discontinuous"]
        ):
            raise ValueError(
                f"`Psi_BC_flag` must be 'continuous', 'discontinuous' or None, but got "
                f"{Psi_BC_flag!r} (Scotty v3's deprecated True meant 'continuous', and "
                "its False meant None)"
            )
        if launch_flag == "vacuum_numerical" and Psi_BC_flag is not None:
            raise ValueError(
                "`launch_flag` = 'vacuum_numerical' integrates from the antenna through "
                "the plasma boundary, with no boundary conditions, so `Psi_BC_flag` "
                f"must be None, but got {Psi_BC_flag!r}. For boundary conditions at "
                "the plasma edge, use `launch_flag` = 'vacuum_analytic'"
            )
        return Psi_BC_flag

    def _unpack_kwargs_for_circular_flux_surfaces(self, **extras):
        # Pop from the caller's kwargs so they are not reported as unused
        extras = extras["extras"]
        B_T_axis = extras.pop("B_T_axis", None)
        B_p_a = extras.pop("B_p_a", None)
        R_axis = extras.pop("R_axis", None)
        minor_radius_a = extras.pop("minor_radius_a", None)

        # TODO -- do we do it like this?
        # either all of them are defined, or none of them are
        # should also insert type hinting and checks
        # `is not None`, not truthiness: B_p_a = 0 (no poloidal field) is valid
        _sum = sum(x is not None for x in (B_T_axis, B_p_a, R_axis, minor_radius_a))
        if _sum not in [0, 4]:
            raise ValueError(f"""
    For circular flux surfaces, all variables must be specified or None, but got:
       - B_T_axis = {B_T_axis}
       - B_p_a = {B_p_a}
       - R_axis = {R_axis}
       - minor_radius_a = {minor_radius_a}
    """)
        elif _sum == 4:
            self.circular_flux_surfaces_flag = True
        else:
            self.circular_flux_surfaces_flag = False

        self.B_T_axis = B_T_axis
        self.B_p_a = B_p_a
        self.R_axis = R_axis
        self.minor_radius_a = minor_radius_a

    def _unpack_kwargs_for_plasma_launch(self, **extras):
        extras = extras["extras"]
        K_plasmaLaunch_cartesian = extras.pop("plasmaLaunch_K_cartesian", np.zeros(3))
        Psi_3D_plasmaLaunch_labframe_cartesian = extras.pop(
            "plasmaLaunch_Psi_3D_lab_cartesian", np.zeros([3, 3])
        )

        if isinstance(K_plasmaLaunch_cartesian, list):
            K_plasmaLaunch_cartesian = np.array(K_plasmaLaunch_cartesian)
        if isinstance(Psi_3D_plasmaLaunch_labframe_cartesian, list):
            Psi_3D_plasmaLaunch_labframe_cartesian = np.array(
                Psi_3D_plasmaLaunch_labframe_cartesian
            )

        _invalid = np.all(K_plasmaLaunch_cartesian == 0) or np.all(
            Psi_3D_plasmaLaunch_labframe_cartesian == 0
        )

        if self.launch_flag == "plasma" and _invalid:
            raise ValueError(f"""
    For plasma launch, `plasmaLaunch_K_cartesian` and `plasmaLaunch_Psi_3D_lab_cartesian`
    must be specified (non-zero) if `launch_flag` = "plasma", but got:
       - launch_flag = {self.launch_flag}
       - plasmaLaunch_K_cartesian = {K_plasmaLaunch_cartesian}
       - plasmaLaunch_Psi_3D_lab_cartesian = {Psi_3D_plasmaLaunch_labframe_cartesian}
    """)

        self.K_plasmaLaunch_cartesian = K_plasmaLaunch_cartesian
        self.Psi_3D_plasmaLaunch_labframe_cartesian = (
            Psi_3D_plasmaLaunch_labframe_cartesian
        )

    def _unpack_unused_kwargs(self, **extras):
        extras_copy = extras["extras"].copy()

        extras_copy.pop("create_magnetic_geometry", None)

        if extras_copy:
            # Measured on the unused keywords themselves (not on `extras`,
            # whose only key is "extras")
            _indent = max(len(max(extras_copy, key=len)), len("keyword")) + 3
            known = _known_argument_names()

            def _suggestion(name: str) -> str:
                close = difflib.get_close_matches(name, known, n=1)
                return f"   (did you mean `{close[0]}`?)" if close else ""

            _msg = "\n".join(
                f"       - {k:<{_indent}} {v!r}{_suggestion(k)}"
                for k, v in extras_copy.items()
            )
            log.warning(
                "Unused keyword arguments, which are not arguments of `beam_me_up` "
                f"and are ignored:\n         {'keyword':<{_indent}} value\n{_msg}"
            )
            removed = sorted(set(extras_copy) & set(REMOVED_ARGUMENTS))
            if removed:
                log.warning(
                    f"{', '.join(removed)}: removed in Scotty 4.0.0, as they did "
                    "nothing (see `scotty.plotting` for figures)"
                )
            v3_names = sorted(set(extras_copy) & set(V3_ONLY_ARGUMENTS))
            if v3_names:
                log.warning(
                    f"{', '.join(v3_names)}: Scotty v3 arguments. To run a v3 script, "
                    "pass its arguments through `scotty.compat.from_v3_kwargs`"
                )

    def _check_methods_given(
        self, find_B_method, density_fit_method, temperature_fit_method
    ) -> None:
        """The sources of the field and the profiles are never guessed: before
        4.0.0, an unset ``find_B_method`` read a TORBEAM topfile, an unset
        ``density_fit_method`` read ``ne.dat`` or guessed the fit from the
        number of parameters, and an unset temperature read ``Te.dat``"""
        if find_B_method is None:
            raise ValueError(
                "`find_B_method` must be given: a field object, or a file "
                'format such as "torbeam" or "omfit", or "analytical"'
            )
        if (
            isinstance(find_B_method, str)
            and find_B_method.lower() in REMOVED_FIND_B_METHODS
        ):
            raise removed_find_B_method_error(find_B_method)
        if density_fit_method is None:
            raise ValueError(
                "`density_fit_method` must be given: a profile object (see "
                "`scotty.profile_fit`), the name of a fit, or "
                '"smoothing-spline-file" to read the file `ne_data_path`'
            )
        if self.relativistic_flag and temperature_fit_method is None:
            raise ValueError(
                "`relativistic_flag=True` needs `temperature_fit_method`: a "
                "profile object, the name of a fit, or "
                '"smoothing-spline-file" to read the file `Te_data_path`'
            )

    @staticmethod
    def _check_analysis_flags(analysis_flag, dbs_analysis_flag):
        for name, value in (
            ("analysis_flag", analysis_flag),
            ("dbs_analysis_flag", dbs_analysis_flag),
        ):
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError(f"`{name}` must be True or False, not {value!r}")
        if dbs_analysis_flag and not analysis_flag:
            raise ValueError(
                "`dbs_analysis_flag=True` needs `analysis_flag=True`: the DBS "
                "analysis uses the beam analysis"
            )
        return bool(analysis_flag), bool(dbs_analysis_flag)

    ##################################################
    #
    # POST-INITIALISATION FUNCTIONS
    #
    ##################################################

    def set_experimental_profiles(self):
        """The parameters of ``"smoothing-spline-file"``: the file and the
        order and smoothing of its spline. The temperature is read only for a
        relativistic run (as in v3, beam-tracing/Scotty#151)"""
        if _reads_file(self.density_fit_method):
            self.density_fit_parameters = [
                self.ne_data_path,
                self.interp_order_ne,
                self.interp_smoothing_ne,
            ]
        if self.relativistic_flag and _reads_file(self.temperature_fit_method):
            self.temperature_fit_parameters = [
                self.Te_data_path,
                self.interp_order_Te,
                self.interp_smoothing_Te,
            ]

    def coordinate_conversions(self):
        r"""Set the positions, wavevectors and Psi of the run in both
        coordinate systems (``*_cartesian`` and ``*_cylindrical``), from those in
        the coordinates of ``geometry``. In cylindrical coordinates, ``K_zeta``
        is the toroidal mode number."""
        if self.cartesian_flag:
            # position vectors
            self.q_launch_cartesian = self.q_launch  # (3,)
            self.q_launch_cylindrical = find_q_labframe_cart_to_cyl(
                self.q_launch_cartesian
            )
            self.q_initial_cartesian = self.q_initial  # (3,)
            self.q_initial_cylindrical = find_q_labframe_cart_to_cyl(
                self.q_initial_cartesian
            )
            self.q_output_cartesian = self.q_output  # (N,3)
            self.q_output_cylindrical = find_q_labframe_cart_to_cyl(
                self.q_output_cartesian.T
            ).T

            # wavevectors
            self.K_launch_cartesian = self.K_launch  # (3,)
            self.K_launch_cylindrical = find_K_labframe_cart_to_cyl(
                self.K_launch_cartesian, self.q_launch_cartesian
            )
            self.K_initial_cartesian = self.K_initial  # (3,)
            self.K_initial_cylindrical = find_K_labframe_cart_to_cyl(
                self.K_initial_cartesian, self.q_initial_cartesian
            )
            self.K_output_cartesian = self.K_output  # (N,3)
            self.K_output_cylindrical = find_K_labframe_cart_to_cyl(
                self.K_output_cartesian.T, self.q_output_cartesian.T
            ).T

            # beam matrices
            if not self.ray_tracing_flag:
                # otherwise the type checker complains
                assert self.Psi_3D_launch_labframe is not None
                assert self.Psi_3D_entry_labframe is not None
                assert self.Psi_3D_initial_labframe is not None
                assert self.Psi_3D_output_labframe is not None
                self.Psi_3D_launch_labframe_cartesian = (
                    self.Psi_3D_launch_labframe
                )  # (3,3)
                self.Psi_3D_launch_labframe_cylindrical = (
                    find_Psi_3D_labframe_cart_to_cyl(
                        self.Psi_3D_launch_labframe_cartesian,
                        self.K_launch_cartesian,
                        self.q_launch_cartesian,
                    )
                )
                self.Psi_3D_entry_labframe_cartesian = (
                    self.Psi_3D_entry_labframe
                )  # (3,3)
                self.Psi_3D_entry_labframe_cylindrical = (
                    find_Psi_3D_labframe_cart_to_cyl(
                        self.Psi_3D_entry_labframe_cartesian,
                        self.K_launch_cartesian,
                        self.q_initial_cartesian,
                    )
                )
                self.Psi_3D_initial_labframe_cartesian = (
                    self.Psi_3D_initial_labframe
                )  # (3,3)
                self.Psi_3D_initial_labframe_cylindrical = (
                    find_Psi_3D_labframe_cart_to_cyl(
                        self.Psi_3D_initial_labframe_cartesian,
                        self.K_initial_cartesian,
                        self.q_initial_cartesian,
                    )
                )
                self.Psi_3D_output_labframe_cartesian = (
                    self.Psi_3D_output_labframe
                )  # (N,3,3)
                self.Psi_3D_output_labframe_cylindrical = (
                    find_Psi_3D_labframe_cart_to_cyl(
                        self.Psi_3D_output_labframe_cartesian,
                        self.K_output_cartesian.T,
                        self.q_output_cartesian.T,
                    )
                )

        else:  # elif not self.cartesian_flag:
            # position vectors
            self.q_launch_cylindrical = self.q_launch  # (3,)
            self.q_launch_cartesian = find_q_labframe_cyl_to_cart(
                self.q_launch_cylindrical
            )
            self.q_initial_cylindrical = self.q_initial  # (3,)
            self.q_initial_cartesian = find_q_labframe_cyl_to_cart(
                self.q_initial_cylindrical
            )
            self.q_output_cylindrical = self.q_output  # (N,3)
            self.q_output_cartesian = find_q_labframe_cyl_to_cart(
                self.q_output_cylindrical.T
            ).T

            # wavevectors
            self.K_launch_cylindrical = self.K_launch  # (3,)
            self.K_launch_cartesian = find_K_labframe_cyl_to_cart(
                self.K_launch_cylindrical, self.q_launch_cylindrical
            )
            self.K_initial_cylindrical = self.K_initial  # (3,)
            self.K_initial_cartesian = find_K_labframe_cyl_to_cart(
                self.K_initial_cylindrical, self.q_initial_cylindrical
            )
            self.K_output_cylindrical = self.K_output  # (N,3)
            self.K_output_cartesian = find_K_labframe_cyl_to_cart(
                self.K_output_cylindrical.T, self.q_output_cylindrical.T
            ).T

            # beam matrices
            if not self.ray_tracing_flag:
                # otherwise the type checker complains
                assert self.Psi_3D_launch_labframe is not None
                assert self.Psi_3D_entry_labframe is not None
                assert self.Psi_3D_initial_labframe is not None
                assert self.Psi_3D_output_labframe is not None
                self.Psi_3D_launch_labframe_cylindrical = (
                    self.Psi_3D_launch_labframe
                )  # (3,3)
                self.Psi_3D_launch_labframe_cartesian = (
                    find_Psi_3D_labframe_cyl_to_cart(
                        self.Psi_3D_launch_labframe_cylindrical,
                        self.K_launch_cylindrical,
                        self.q_launch_cylindrical,
                    )
                )
                self.Psi_3D_entry_labframe_cylindrical = (
                    self.Psi_3D_entry_labframe
                )  # (3,3)
                # The entry is still in vacuum, so K there is the launched K,
                # but its cylindrical components are those at the entry, not
                # at the launch point (they differ when zeta changes)
                self.Psi_3D_entry_labframe_cartesian = find_Psi_3D_labframe_cyl_to_cart(
                    self.Psi_3D_entry_labframe_cylindrical,
                    find_K_labframe_cart_to_cyl(
                        self.K_launch_cartesian, self.q_initial_cartesian
                    ),
                    self.q_initial_cylindrical,
                )
                self.Psi_3D_initial_labframe_cylindrical = (
                    self.Psi_3D_initial_labframe
                )  # (3,3)
                self.Psi_3D_initial_labframe_cartesian = (
                    find_Psi_3D_labframe_cyl_to_cart(
                        self.Psi_3D_initial_labframe_cylindrical,
                        self.K_initial_cylindrical,
                        self.q_initial_cylindrical,
                    )
                )
                self.Psi_3D_output_labframe_cylindrical = (
                    self.Psi_3D_output_labframe
                )  # (N,3,3)
                self.Psi_3D_output_labframe_cartesian = (
                    find_Psi_3D_labframe_cyl_to_cart(
                        self.Psi_3D_output_labframe_cylindrical,
                        self.K_output_cylindrical.T,
                        self.q_output_cylindrical.T,
                    )
                )

        self.K_output_magnitude = find_K_magnitude(
            True, *self.K_output_cartesian.T, self.q_output_cartesian.T[0]
        )  # (N,) # type: ignore
        self.K_hat_output_cartesian = (
            self.K_output_cartesian / self.K_output_magnitude[:, np.newaxis]
        )

        if self.ray_tracing_flag:
            self.Psi_3D_launch_labframe_cartesian = None
            self.Psi_3D_launch_labframe_cylindrical = None
            self.Psi_3D_entry_labframe_cartesian = None
            self.Psi_3D_entry_labframe_cylindrical = None
            self.Psi_3D_initial_labframe_cartesian = None
            self.Psi_3D_initial_labframe_cylindrical = None
            self.Psi_3D_output_labframe_cartesian = None
            self.Psi_3D_output_labframe_cylindrical = None


##################################################
#
# OUTPUT CHECKS AFTER RAY AND BEAM TRACING
#
##################################################


def check_H_output(H_arr: ArrayLike, H_tol: float = 1e-2) -> None:
    r"""Check that the dispersion relation holds along the ray: H should be
    zero. Raises a `ValueError` if max(abs(H)) > 0.1 (for example a ray on
    the wrong root, which is at about 1), and warns if it is above
    ``H_tol``."""
    log.debug("Checking `H` output along the ray (it should be zero)")

    H_max = np.max(np.abs(H_arr))
    # On-surface drift is about 10 rtol, so about 0.01 at the default. A ray on the
    # wrong root is at about 1 (beam-tracing/Scotty#169)
    if H_max > 0.1:
        raise ValueError(
            f"`H_output` is unphysically large! `max(|H_output|)` = {H_max}"
        )
    elif H_max > H_tol:
        log.warning(
            f"`H_output` is unphysically large! `max(|H_output|)` = {H_max} but `H_tol` = {H_tol}"
        )


def check_Psi_g_output(
    e_imag: ArrayLike,
    e_real: ArrayLike,
    tol: float = 0.1,
    fraction_tol: float = 0.1,
) -> None:
    r"""Warns if :math:`\Psi \cdot \mathbf{g} = \mathrm{d}\mathbf{K}/\mathrm{d}\tau`
    fails along much of the beam, i.e. if either residual of
    `scotty.fun_general.find_Psi_g_residuals` is above `tol` at more than
    `fraction_tol` of the points.

    At the default solver tolerances the residuals are a few 1e-3 or less on
    correct beams, and about 0.3 or more where Psi has drifted off the beam
    (e.g. a density that isn't differentiable). This checks that Psi is
    consistent with the ray, not that the beam model is valid: past a cut-off
    at almost exactly normal incidence, for instance, the beam is meaningless
    but the residuals are small again. A few points can exceed `tol`
    on a correct beam, hence the fraction rather than the maximum: near a
    cut-off at close to normal incidence, where K is small, and at the last
    point if it is just outside the plasma, where the density gradient jumps
    to zero."""
    log.debug("Checking Psi . g = dK/dtau along the beam")

    e_imag, e_real = np.asarray(e_imag), np.asarray(e_real)
    # NaN where Psi or g vanish; such points can't be checked
    failed = np.nan_to_num(np.maximum(e_imag, e_real), nan=0.0) > tol
    fraction = np.count_nonzero(failed) / failed.size
    if fraction > fraction_tol:
        log.warning(
            f"Psi . g differs from dK/dtau by more than {tol} (relative) at "
            f"{fraction:.0%} of the points along the beam (median residuals: "
            f"imaginary part {np.nanmedian(e_imag):.2e}, real part "
            f"{np.nanmedian(e_real):.2e}). Psi is then not the gradient of K along "
            "the beam, so the beam widths and curvatures can't be trusted. Likely "
            "causes: a density or field that isn't smooth enough (e.g. piecewise "
            "linear), or tolerances too loose for this case (`rtol`, `atol`)"
        )
