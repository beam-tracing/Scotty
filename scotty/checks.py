import logging
from multiprocessing.managers import ValueProxy
from pathlib import Path
from scotty.compat import V3_ONLY_ARGUMENTS
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
    MagneticField_Cylindrical,
    MagneticField_Cartesian,
    InterpolatedField_Cylindrical,
    InterpolatedField_Cartesian,
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
VALID_LAUNCH_FLAGS = Literal["plasma", "vacuum"]
VALID_BOUNDARY_FLAGS = Optional[Literal["continuous", "discontinuous"]]
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


class Parameters:
    # Initialisating here to stop type checker from complaining
    geometry: VALID_GEOMETRIES
    cartesian_flag: bool
    launch_flag: VALID_LAUNCH_FLAGS
    boundary_flag: VALID_BOUNDARY_FLAGS
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
        launch_beam_width: float,
        launch_beam_curvature: float,
        q_launch: FloatArray,
        mode_flag_launch: VALID_LAUNCH_MODE_FLAGS,
        # Solver settings and finite-difference parameters
        launch_flag: VALID_LAUNCH_FLAGS,
        boundary_flag: VALID_BOUNDARY_FLAGS,
        relativistic_flag: bool,
        auto_delta_sign: bool,
        deltas_cylindrical: FloatArray,
        deltas_cartesian: FloatArray,
        len_tau: int,
        rtol: float,
        atol: float,
        poloidal_flux_enter: float,
        poloidal_flux_zero_density: float,
        poloidal_flux_zero_temperature: float,
        # Data input and output arguments
        magnetic_data_path: Union[str, Path],
        ne_data_path: Union[str, Path],
        Te_data_path: Union[str, Path],
        input_filename_suffix: str,
        output_path: Union[str, Path],
        output_filename_suffix: str,
        shot: Optional[int],
        equil_time: Optional[Union[int, float]],
        # Interpolation settings
        find_B_method: Union[str, MagneticField_Cylindrical, MagneticField_Cartesian],
        density_fit_method: Optional[Union[str, ProfileFitLike]],
        density_fit_parameters: Optional[Sequence],
        temperature_fit_method: Optional[Union[str, ProfileFitLike]],
        temperature_fit_parameters: Optional[Sequence],
        interp_order: Union[str, int],
        interp_smoothing: int,
        # Plotting flags
        figure_flag: bool,
        further_analysis_flag: bool,
        detailed_analysis_flag: bool,
        # Additional flags
        benchmarking_flag: bool,
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

        # TORBEAM antenna angles are anti-clockwise from negative X-axis,
        # so we need to rotate the toroidal angle by pi. This will take
        # care of the direction of the beam. The poloidal angle is also
        # reversed from its usual sense, so we can just flip it by adding
        # a minus sign
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
            "launch_beam_width", launch_beam_width
        )
        self.launch_beam_curvature = launch_beam_curvature
        self.q_launch = q_launch
        self.mode_flag_launch = self._check_mode_flag(mode_flag_launch)

        ##################################################
        #
        # Solver settings and finite-difference parameters
        #
        ##################################################

        self.benchmarking_flag = benchmarking_flag
        self.ray_tracing_flag = ray_tracing_flag
        self.launch_flag = launch_flag
        self.boundary_flag = boundary_flag
        self.relativistic_flag = relativistic_flag
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
        self.magnetic_data_path = self._check_data_path("magnetic", magnetic_data_path)
        self.interp_order_magnetic_data_str, self.interp_order_magnetic_data_int = (
            self._check_interp_order("magnetic", interp_order)
        )

        self.ne_data_path = self._check_data_path("ne", ne_data_path)
        _, self.interp_order_ne_data = self._check_interp_order(
            "ne", interp_order
        )  # TODO -- is this needed? or same interp_order for B and ne?
        self.density_fit_parameters = density_fit_parameters
        self.density_fit_method = density_fit_method

        self.Te_data_path = self._check_data_path("Te", Te_data_path)
        self.interp_order_Te_data = (
            None  # TODO -- not implemented yet? implement together with temp. profile
        )
        self.temperature_fit_parameters = temperature_fit_parameters
        self.temperature_fit_method = temperature_fit_method

        self.interp_smoothing = interp_smoothing
        self.shot = shot
        self.equil_time = equil_time

        self.input_filename_suffix = input_filename_suffix
        self.output_path = self._check_data_path("output", output_path)
        self.output_filename_suffix = output_filename_suffix

        self.figure_flag = figure_flag
        self.further_analysis_flag = further_analysis_flag
        self.detailed_analysis_flag = detailed_analysis_flag

        # TODO should check this works properly again
        self._check_mode_flag_launch_and_launch_flag(
            mode_flag_launch=mode_flag_launch, launch_flag=launch_flag
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
    # PRIMARY INPUT CHECKS (for beam_me_up_3D)
    # For data passed at first call of beam_me_up_3D
    #
    # TODO:
    #    1) Need to put data pathway stuff here
    #
    #    2a) Put the density_fit_parameters and
    #        density_fit_method stuff here
    #
    #    2b) Put ne_data checks here:
    #        - rho must be sorted
    #        - ne must be sorted???
    #        - must polflux_zero_density be less than
    #          the last polflux coord in ne.dat?
    #
    #    3) Need to put temperature data stuff here
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
        if value <= 0:
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
        poloidal_flux_enter: float,
        poloidal_flux_zero_density: float,
        poloidal_flux_zero_temperature: float,
        relativistic_flag: bool,
    ) -> Tuple[float, float, float]:
        self._check_positive("poloidal_flux_enter", poloidal_flux_enter)
        self._check_positive("poloidal_flux_zero_density", poloidal_flux_zero_density)
        self._check_positive(
            "poloidal_flux_zero_temperature", poloidal_flux_zero_temperature
        )
        if poloidal_flux_zero_density < poloidal_flux_enter:
            raise ValueError(
                f"`poloidal_flux_zero_density` must be greater than or equal to `poloidal_flux_enter`, but got `poloidal_flux_zero_density` = {poloidal_flux_zero_density} and `poloidal_flux_enter` = {poloidal_flux_enter}"
            )
        if poloidal_flux_zero_temperature < poloidal_flux_enter:
            raise ValueError(
                f"`poloidal_flux_zero_temperature` must be greater than or equal to `poloidal_flux_enter`, but got `poloidal_flux_zero_temperature` = {poloidal_flux_zero_temperature} and `poloidal_flux_enter` = {poloidal_flux_enter}"
            )
        if (
            poloidal_flux_zero_density != poloidal_flux_zero_temperature
            and relativistic_flag
        ):
            log.warning(
                f"`poloidal_flux_zero_density` ({poloidal_flux_zero_density}) and `poloidal_flux_zero_temperature` ({poloidal_flux_zero_temperature}) are set to different values. This may cause issues because `relativistic_flag` = {relativistic_flag}"
            )
        return (
            poloidal_flux_enter,
            poloidal_flux_zero_density,
            poloidal_flux_zero_temperature,
        )

    def _check_data_path(
        self, name: Literal["magnetic", "ne", "Te", "output"], path: Union[str, Path]
    ) -> Path:
        path = Path(path)
        if name in ["magnetic", "ne", "Te"] and not path.is_dir():
            raise ValueError(f"`{name}_data_path` must be a valid directory")
        elif name in ["output"] and not path.is_dir():
            print(f"`{name}_data_path` does not exist. Creating the folder now")
            os.makedirs(path)

        return path

    def _check_interp_order(self, name: str, order: Union[str, int]) -> Tuple[str, int]:
        _valid_interp_orders_int = {1: "linear", 3: "cubic", 5: "quintic"}
        _valid_interp_orders_str = dict(
            (v, k) for k, v in _valid_interp_orders_int.items()
        )

        if order in _valid_interp_orders_str:
            new_orders = order, _valid_interp_orders_str[order]
        elif order in _valid_interp_orders_int:
            new_orders = _valid_interp_orders_int[order], order
        else:
            new_orders = "quintic", 5
            log.warning(f"""
    `interp_order` for `{name}` data must be one of {list(_valid_interp_orders_int) + list(_valid_interp_orders_str)}, but got {order}
    
    Ignoring user-specified argument and setting `interp_order` = `quintic`. This may or may not cause issues
    """)

        return new_orders

    def _check_mode_flag_launch_and_launch_flag(
        self, mode_flag_launch: VALID_LAUNCH_MODE_FLAGS, launch_flag: VALID_LAUNCH_FLAGS
    ):
        if launch_flag == "vacuum":
            pass
        elif launch_flag == "plasma" and mode_flag_launch in ["O", "X"]:
            raise ValueError(
                f"`mode_flag` (at launch) must be '1' or '-1' if `launch_flag` = 'plasma', but got `mode_flag` = {mode_flag_launch} and `launch_flag` = {launch_flag}"
            )
        else:
            raise ValueError(
                f"`launch_flag` must be one of {get_args(VALID_LAUNCH_FLAGS)}, but got {launch_flag}"
            )

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
            _indent = max(len(max(extras, key=len)), len("keyword")) + 3
            _printmsg = "\n".join(
                f"       - {k:<{_indent}} {v}" for k, v in extras_copy.items()
            )

            print(f"""
    There are unsued keyword arguments that do not correspond to any function:
         {"keyword":<{_indent}} value \n{_printmsg}
    """)
            v3_names = sorted(set(extras_copy) & set(V3_ONLY_ARGUMENTS))
            if v3_names:
                log.warning(
                    f"{', '.join(v3_names)}: Scotty v3 arguments. To run a v3 script, "
                    "pass its arguments through `scotty.compat.from_v3_kwargs`"
                )

    ##################################################
    #
    # POST-INITIALISATION FUNCTIONS
    #
    ##################################################

    def set_experimental_profiles(self):
        if (self.density_fit_parameters is None) and (
            self.density_fit_method in [None, "smoothing-spline-file"]
        ):
            self.ne_filename = self.ne_data_path / f"ne{self.input_filename_suffix}.dat"
            self.density_fit_parameters = [
                self.ne_filename,
                self.interp_order_ne_data,
                self.interp_smoothing,
            ]
        else:
            self.ne_filename = None

    def coordinate_conversions(self):
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
                self.Psi_3D_entry_labframe_cartesian = find_Psi_3D_labframe_cyl_to_cart(
                    self.Psi_3D_entry_labframe_cylindrical,
                    self.K_launch_cylindrical,
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
# INPUT CHECK BEFORE RAY TRACING
#
##################################################


def check_H_output(H_arr: ArrayLike, H_tol: float = 1e-2) -> None:
    log.debug("Checking `H` output along the ray (it should be zero)")

    H_max = np.abs(np.max(H_arr))
    if H_max > 1.0:
        raise ValueError(
            f"`H_output` is unphysically large! `|max(H_output)|` = {H_max}"
        )
    elif H_max > H_tol:
        log.warning(
            f"`H_output` is unphysically large! `|max(H_output)|` = {H_max} but `H_tol` = {H_tol}"
        )


def check_Psi_output(
    Psi_xg_output: FloatArray,
    Psi_yg_output: FloatArray,
    Psi_gg_output: FloatArray,
    Psi_tol: float = 1e-3,
) -> None:
    log.debug("Checking Im(Psi) g-components along ray (it should be zero)")

    max_xg = np.abs(np.max(Psi_xg_output))
    max_yg = np.abs(np.max(Psi_yg_output))
    max_gg = np.abs(np.max(Psi_gg_output))
    if max_xg > Psi_tol:
        log.warning(
            f"`Psi_xg` is unphysically large! `|max(Psi_xg)|` = {max_xg} but `Psi_tol` = {Psi_tol}"
        )
    if max_yg > Psi_tol:
        log.warning(
            f"`Psi_yg` is unphysically large! `|max(Psi_yg)|` = {max_yg} but `Psi_tol` = {Psi_tol}"
        )
    if max_gg > Psi_tol:
        log.warning(
            f"`Psi_gg` is unphysically large! `|max(Psi_gg)|` = {max_gg} but `Psi_tol` = {Psi_tol}"
        )
