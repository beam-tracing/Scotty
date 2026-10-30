# -*- coding: utf-8 -*-
# Copyright 2018 - 2023, Valerian Hall-Chen and the Scotty contributors
# SPDX-License-Identifier: GPL-3.0

"""
The entry point of Scotty: `beam_me_up` traces the ray, propagates the beam
and analyses the result, saving everything to one output file.

Notes
~~~~~

- ``geometry`` is ``"cylindrical"`` (axisymmetric fields; :math:`K_\\zeta` is
  conserved, and :math:`K^2 = K_R^2 + (K_\\zeta / R)^2 + K_Z^2`) or
  ``"cartesian"`` (general 3D fields).
- ``launch_flag="vacuum"`` launches the beam in vacuum, propagates it to the
  plasma and applies the boundary condition there (``boundary_flag``:
  ``"continuous"``, ``"discontinuous"`` or ``None``). ``launch_flag="plasma"``
  starts the beam inside the plasma, with its :math:`K` and :math:`\\Psi`
  given directly.
- Positions along the way: *launch* is where the beam starts in vacuum,
  *entry* is where it reaches the plasma boundary, and *initial* is the first
  point in the plasma, where the solver starts (e.g. ``q_launch``,
  ``Psi_3D_entry_labframe``, ``q_initial``).
- ``ray_tracing_flag=True`` traces only the ray, with no beam.
- Scripts written for Scotty v3 can be converted with `scotty.compat`.

Output
~~~~~~

An `xarray.DataTree` with the groups ``inputs``, ``solver_output``,
``analysis`` and ``dbs_analysis``, saved as ``scotty_output<suffix>.h5``.

Coordinates
~~~~~~~~~~~

- ``X, Y, Z`` - Lab Cartesian coordinates
- ``R, zeta, Z`` - Lab cylindrical coordinates
- ``x, y, g`` - Beam coordinates
- ``u1, u2, b`` - Field-aligned coordinates

Abbreviations
~~~~~~~~~~~~~

- ``bs`` - backscattered
- ``loc`` - localisation (``loc_b``, ``loc_p``, ``loc_r``, ``loc_m``)
- ``ne`` - equilibrium electron density
- ``dt`` - datatree (stores data)

Angles
~~~~~~

- ``theta`` - angle between g and u1, small when mismatch is small
- ``theta_m`` - mismatch angle, angle between u1 and K

Units
~~~~~

- SI units
- Distance in m
- Angles in rad
- Temperature in keV
- Electron cyclotron frequency positive
"""

import datetime
import logging
import numpy as np
import pathlib
from scotty.analysis import basic_analysis, dbs_analysis
from scotty.beam_solver import beam_tracing
from scotty.checks import (
    VALID_GEOMETRIES,
    VALID_LAUNCH_FLAGS,
    VALID_LAUNCH_MODE_FLAGS,
    VALID_BOUNDARY_FLAGS,
    Parameters,
)
from scotty.data_io import inputs_to_dicts, solver_output_to_dicts, dict_to_dataset
from scotty.geometry import (
    MagneticField_Cylindrical,
    MagneticField_Cartesian,
    create_magnetic_geometry,
)
from scotty.hamiltonian import initialise_hamiltonians, assign_hamiltonians
from scotty.launch import (
    find_plasma_entry_position,
    find_auto_delta_signs,
    find_plasma_entry_parameters,
)
from scotty.logger import config_logger, arr2str
from scotty.profile_fit import ProfileFitLike, profile_fit
from scotty.ray_solver import ray_tracing
from scotty.typing import FloatArray, PathLike
from scotty._version import __version__
from typing import Optional, Sequence, Tuple, Union, Literal, cast, overload
import uuid
import xarray as xr


def beam_me_up(
    # Main parameters
    geometry: VALID_GEOMETRIES,
    poloidal_launch_angle_Torbeam: float,
    toroidal_launch_angle_Torbeam: float,
    launch_freq_GHz: float,
    launch_beam_width: float,
    launch_beam_curvature: float,
    launch_position: FloatArray,
    mode_flag: VALID_LAUNCH_MODE_FLAGS,
    # Solver settings and finite-difference parameters
    launch_flag: VALID_LAUNCH_FLAGS,
    boundary_flag: VALID_BOUNDARY_FLAGS,
    relativistic_flag: bool = False,  # includes relativistic corrections to electron mass when set to True
    resonance_flag: bool = True,  # stop the ray at a cyclotron (or, in X-mode, upper-hybrid) resonance
    auto_delta_sign: bool = True,  # For flipping signs to maintain forward difference
    delta_X: float = 1e-4,  # in the same units as data_X_coord
    delta_Y: float = 1e-4,  # in the same units as data_Y_coord
    delta_R: float = 1e-4,
    delta_Z: float = 1e-4,  # in the same units as data_Z_coord
    delta_K_X: float = 1e-1,  # in the same units as K_X
    delta_K_Y: float = 1e-1,  # in the same units as K_Y
    delta_K_R: float = 1e-1,
    delta_K_zeta: float = 1e-1,
    delta_K_Z: float = 1e-1,  # in the same units as K_Z
    len_tau: int = 102,
    rtol: float = 1e-3,  # for solve_ivp of the ray/beam solvers
    atol: float = 1e-6,  # for solve_ivp of the ray/beam solvers
    poloidal_flux_enter: float = 1.0,
    poloidal_flux_zero_density: float = 1.0,  # When polflux >= poloidal_flux_zero_density, Scotty sets density = 0
    poloidal_flux_zero_temperature: float = 1.0,  # Temperature analogue of poloidal_flux_zero_density
    # Data input and output arguments
    magnetic_data_path=pathlib.Path("."),
    ne_data_path=pathlib.Path("."),
    Te_data_path=pathlib.Path("."),
    input_filename_suffix: str = "",
    output_path=pathlib.Path("."),
    output_filename_suffix: str = "",
    shot: Optional[int] = None,
    equil_time: Optional[Union[float, int]] = None,
    # Interpolation settings
    find_B_method: Union[
        str, MagneticField_Cylindrical, MagneticField_Cartesian
    ] = "omfit",
    density_fit_method: Optional[Union[str, ProfileFitLike]] = None,
    density_fit_parameters: Optional[Sequence] = None,
    temperature_fit_method: Optional[Union[str, ProfileFitLike]] = None,
    temperature_fit_parameters: Optional[Sequence] = None,
    interp_order: Union[str, int] = 5,  # For the 3D interpolation functions
    interp_smoothing: int = 5,  # For the 3D interpolation functions (specifically, make_fit)
    # Logging flags
    console_log_level: Union[
        str, int
    ] = "INFO",  # For returning log messages on console
    file_log_level: Optional[
        Union[str, int]
    ] = None,  # For returning log messages on a log file
    # Plotting flags
    figure_flag: bool = True,
    further_analysis_flag: bool = False,
    detailed_analysis_flag: bool = False,
    # TODO -- need to put individual flags for each plot
    # Additional flags
    benchmarking_flag: bool = False,
    ray_tracing_flag: bool = False,
    return_dt_field: bool = False,  # For returning the datatree, field class
    # Keeping the extra kwargs for parsing later
    **kwargs,
) -> Union[
    xr.DataTree, Tuple[xr.DataTree, MagneticField_Cylindrical | MagneticField_Cartesian]
]:

    """Trace a beam and compute its DBS localisation.

    Scripts written for Scotty v3 use arguments that v4 renamed or removed
    (e.g. ``Psi_BC_flag``, ``vacuumLaunch_flag``), and miss the required
    ``geometry``, ``launch_flag`` and ``boundary_flag``. Convert them with
    ``beam_me_up(**scotty.compat.from_v3_kwargs(kwargs))``.
    """
    # INSERT DOC STRING HERE

    ##################################################
    #
    # INITIALISATION ROUTINE
    #
    ##################################################
    params = Parameters(
        # Main parameters
        geometry=geometry,
        poloidal_launch_angle_Torbeam=poloidal_launch_angle_Torbeam,
        toroidal_launch_angle_Torbeam=toroidal_launch_angle_Torbeam,
        launch_freq_GHz=launch_freq_GHz,
        launch_beam_width=launch_beam_width,
        launch_beam_curvature=launch_beam_curvature,
        q_launch=launch_position,
        mode_flag_launch=mode_flag,
        # Solver settings and finite-difference parameters
        launch_flag=launch_flag,
        boundary_flag=boundary_flag,
        relativistic_flag=relativistic_flag,
        resonance_flag=resonance_flag,
        auto_delta_sign=auto_delta_sign,
        deltas_cylindrical=np.array(
            [delta_R, delta_Z, delta_K_R, delta_K_zeta, delta_K_Z], dtype=float
        ),
        deltas_cartesian=np.array(
            [delta_X, delta_Y, delta_Z, delta_K_X, delta_K_Y, delta_K_Z], dtype=float
        ),
        len_tau=len_tau,
        rtol=rtol,
        atol=atol,
        poloidal_flux_enter=poloidal_flux_enter,
        poloidal_flux_zero_density=poloidal_flux_zero_density,
        poloidal_flux_zero_temperature=poloidal_flux_zero_temperature,
        # Data input and output arguments
        magnetic_data_path=magnetic_data_path,
        ne_data_path=ne_data_path,
        Te_data_path=Te_data_path,
        input_filename_suffix=input_filename_suffix,
        output_path=output_path,
        output_filename_suffix=output_filename_suffix,
        shot=shot,
        equil_time=equil_time,
        # Interpolation settings
        find_B_method=find_B_method,
        density_fit_method=density_fit_method,
        density_fit_parameters=density_fit_parameters,
        temperature_fit_method=temperature_fit_method,
        temperature_fit_parameters=temperature_fit_parameters,
        interp_order=interp_order,
        interp_smoothing=interp_smoothing,
        # Plotting flags
        figure_flag=figure_flag,
        further_analysis_flag=further_analysis_flag,
        detailed_analysis_flag=detailed_analysis_flag,
        # Additional flags
        benchmarking_flag=benchmarking_flag,
        ray_tracing_flag=ray_tracing_flag,
        # return_dt_field = return_dt_field,
        # Extra kwargs for parsing
        **kwargs,
    )

    # Setting up the logger
    config_logger(
        console_log_level,
        file_log_level,
        params.output_path,
        params.output_filename_suffix,
    )
    log = logging.getLogger(__name__)
    log.debug(f"Saved and validated launch parameters")
    log.debug(f"Initialised logger")

    log.info(f"""\n
    ##################################################
    #
    # STARTING ROUTINE
    #
    ##################################################
    #
    # Beam trace me up, Scotty!
    # Version {__version__}
    # Run ID: {uuid.uuid4()}
    #
    ##################################################
    #
    # Starting run for:
    #   - Geometry = {params.geometry}
    #   - Pol. launch angle = {params.poloidal_launch_angle_deg_Torbeam} deg
    #   - Tor. launch angle = {params.toroidal_launch_angle_deg_Torbeam} deg
    #   - Launch frequency = {params.launch_frequency_GHz} GHz
    #   - Launch beam width = {params.launch_beam_width} m
    #   - Launch beam curvature = {params.launch_beam_curvature} m^-1
    #   - Launch position {"[R,zeta,Z]" if params.geometry == "cylindrical" else "[X,Y,Z]"} = {arr2str(params.q_launch)}
    #   - Mode flag = {params.mode_flag_launch}
    #
    #   - find_B_method = {params.find_B_method}
    #   - launch_flag = {params.launch_flag}
    #   - boundary flag = {params.boundary_flag}
    #   - relativistic flag = {params.relativistic_flag}
    #   - figure flag = {params.figure_flag}
    #   - further_analysis flag = {params.further_analysis_flag}
    #   - detailed_analysis flag = {params.detailed_analysis_flag}
    #
    ##################################################
    """)

    log.debug(f"Setting experimental profiles for density")
    params.set_experimental_profiles()

    # UUsing the electron density data, create a spline fit as a
    # function of poloidal flux, i.e.:
    # density_fit(poloidal_flux) = electron_density
    density_fit = make_fit(
        method=params.density_fit_method,
        poloidal_flux_zero_density=params.poloidal_flux_zero_density,
        parameters=params.density_fit_parameters,
        filename=params.ne_filename,
    )

    # Using the temperature data, create a fit as a function of poloidal
    # flux, i.e. temperature_fit(poloidal_flux) = temperature. Only used for
    # relativistic corrections, as in v3 (beam-tracing/Scotty#151)
    if params.relativistic_flag:
        temperature_fit = make_fit(
            method=params.temperature_fit_method,
            poloidal_flux_zero_density=params.poloidal_flux_zero_temperature,
            parameters=params.temperature_fit_parameters,
            filename=params.Te_filename,
            quantity="temperature",
        )
    else:
        temperature_fit = None

    # Analytical field parameters can be passed directly to beam_me_up (as in
    # v3) or nested in `create_magnetic_geometry`; the nested ones take priority
    field_kwargs = {
        key: value
        for key, value in {
            "B_T_axis": params.B_T_axis,
            "R_axis": params.R_axis,
            "minor_radius_a": params.minor_radius_a,
            "B_p_a": params.B_p_a,
        }.items()
        if value is not None
    }
    field_kwargs.update(kwargs.get("create_magnetic_geometry", {}))

    # If the user has already pre-loaded a field, use it
    # Otherwise, create a new one from the specified data pathway
    field = create_magnetic_geometry(
        geometry=params.geometry,
        find_B_method=find_B_method,  # not from params because we only store the string/name, but find_B_method is probably already a MagField class
        interp_order_str=params.interp_order_magnetic_data_str,
        interp_order_int=params.interp_order_magnetic_data_int,
        interp_smoothing=params.interp_smoothing,
        magnetic_data_path=params.magnetic_data_path,
        input_filename_suffix=params.input_filename_suffix,
        shot=params.shot,
        equil_time=params.equil_time,
        **field_kwargs,
    )

    log.info(f"""\n
    ##################################################
    #
    # LAUNCH ROUTINE
    #
    ##################################################
    """)

    # Calculating the plasma entry position and auto_delta_sign
    params.q_initial = find_plasma_entry_position(
        poloidal_launch_angle_deg_Torbeam=params.poloidal_launch_angle_deg_Torbeam,
        toroidal_launch_angle_deg_Torbeam=params.toroidal_launch_angle_deg_Torbeam,
        q_launch=params.q_launch,
        launch_flag=params.launch_flag,
        field=field,
        poloidal_flux_enter=params.poloidal_flux_enter,
        boundary_adjust=1e-6,
    )

    # Setting the delta signs
    spatial_deltas = find_auto_delta_signs(
        auto_delta_sign=params.auto_delta_sign,
        q_initial=params.q_initial,
        deltas=params.deltas[:-3],
        field=field,
    )

    if params.geometry == "cylindrical":
        params.delta_R, params.delta_Z = spatial_deltas
    elif params.geometry == "cartesian":
        params.delta_X, params.delta_Y, params.delta_Z = spatial_deltas
    # The Hamiltonians below take their spacings from `params.deltas`, so
    # update its spatial part as well
    params.deltas = np.concatenate([spatial_deltas, params.deltas[-3:]])

    # Initialises the Hamiltonian H for `mode_flag`s +1 and -1
    (
        hamiltonian_pos1,
        hamiltonian_neg1,
    ) = initialise_hamiltonians(
        launch_angular_freq=params.launch_angular_frequency,
        deltas=params.deltas,
        field=field,
        density_fit=density_fit,
        temperature_fit=temperature_fit,
    )

    # Calculating the plasma entry parameters
    (
        params.K_launch,
        params.K_initial,
        params.Psi_3D_launch_labframe,
        params.Psi_3D_entry_labframe,
        params.Psi_3D_initial_labframe,
        params.distance_from_launch_to_entry,
        params.e_hat_initial,
        params.mode_flag_initial,
        params.mode_index,
    ) = find_plasma_entry_parameters(
        ray_tracing_flag=params.ray_tracing_flag,
        launch_flag=params.launch_flag,
        boundary_flag=params.boundary_flag,
        mode_flag_launch=params.mode_flag_launch,
        poloidal_launch_angle_deg_Torbeam=params.poloidal_launch_angle_deg_Torbeam,
        toroidal_launch_angle_deg_Torbeam=params.toroidal_launch_angle_deg_Torbeam,
        q_launch=params.q_launch,
        q_initial=params.q_initial,
        launch_beam_width=params.launch_beam_width,
        launch_beam_curvature=params.launch_beam_curvature,
        field=field,
        K_plasmaLaunch_cartesian=params.K_plasmaLaunch_cartesian,
        Psi_3D_plasmaLaunch_labframe_cartesian=params.Psi_3D_plasmaLaunch_labframe_cartesian,
        hamiltonian_pos1=hamiltonian_pos1,
        hamiltonian_neg1=hamiltonian_neg1,
        tol_H=1e-2,
        tol_O_mode_polarisation=0.25,
    )

    # Assigning the correct Hamiltonian
    (
        hamiltonian,
        hamiltonian_other,
    ) = assign_hamiltonians(
        mode_flag_initial=params.mode_flag_initial,
        hamiltonian_pos1=hamiltonian_pos1,
        hamiltonian_neg1=hamiltonian_neg1,
        q_initial=params.q_initial,
        K_initial=params.K_initial,
        tol_H=1e-5,
    )

    log.info(f"""\n
    ##################################################
    #
    # RAY TRACING ROUTINE
    #
    ##################################################
    """)

    ray_tracing_result = ray_tracing(
        q_initial=params.q_initial,
        K_initial=params.K_initial,
        poloidal_flux_enter=params.poloidal_flux_enter,
        hamiltonian=hamiltonian,
        ray_tracing_flag=params.ray_tracing_flag,
        rtol=params.rtol,
        atol=params.atol,
        len_tau=params.len_tau,
        resonance_flag=params.resonance_flag,
        # tau_max = 1e5,
    )

    if (
        params.ray_tracing_flag
    ):  # if ray_tracing_flag, then ray_tracing_result is 2-tuple of (tau_arr, q_K_arrs)
        (
            params.solver_status,
            params.solver_nfev,
            params.solver_duration,
            params.tau_output,
            q_K_output,
        ) = cast(
            Tuple[Literal[1, 0, -1], int, float, FloatArray, FloatArray],
            ray_tracing_result,
        )
        params.q_output = q_K_output[:3].T  # (N, 3)
        params.K_output = q_K_output[3:].T  # (N,3 )
        params.Psi_3D_output_labframe = None
    else:  # otherwise just take the tau_array with tau of terminating event and pass into beam-tracing
        _, _, _, tau_points, tau_terminating_event = cast(
            Tuple[Literal[1, 0, -1], int, float, FloatArray, float], ray_tracing_result
        )

        log.info(f"""\n
    ##################################################
    #
    # BEAM TRACING ROUTINE
    #
    ##################################################
    """)

        (
            params.solver_status,
            params.solver_nfev,
            params.solver_duration,
            params.tau_output,  # (N,)
            params.q_output,  # (N, 3)
            params.K_output,  # (N, 3)
            params.Psi_3D_output_labframe,  # (N, 3, 3)
        ) = beam_tracing(
            tau_leave=tau_terminating_event,
            tau_points=tau_points,
            q_initial=params.q_initial,
            K_initial=params.K_initial,
            Psi_3D_initial_labframe=params.Psi_3D_initial_labframe,
            hamiltonian=hamiltonian,
            rtol=params.rtol,
            atol=params.atol,
        )

    # Coordinate conversions for later
    params.coordinate_conversions()

    # Once the ray/beam-tracing is complete, save the data from params
    # as xarray datasets
    inputs_ds = dict_to_dataset(*inputs_to_dicts(params, field))
    solver_output_ds = dict_to_dataset(*solver_output_to_dicts(params))

    # Then do basic analysis (i.e. no DBS) and also DBS analysis and
    # convert both to xarray datasets
    analysis_ds = dict_to_dataset(
        *basic_analysis(
            inputs=inputs_ds,
            solver_output=solver_output_ds,
            hamiltonian=hamiltonian,
            hamiltonian_other=hamiltonian_other,
            field=field,
            density_fit=density_fit,
            temperature_fit=temperature_fit,
        )
    )

    dbs_analysis_ds = dict_to_dataset(
        *dbs_analysis(
            inputs=inputs_ds,
            solver_output=solver_output_ds,
            analysis=analysis_ds,
            hamiltonian=hamiltonian,
            hamiltonian_other=hamiltonian_other,
            field=field,
            density_fit=density_fit,
            temperature_fit=temperature_fit,
        )
    )

    dt = xr.DataTree.from_dict(
        {
            "inputs": inputs_ds,
            "solver_output": solver_output_ds,
            "analysis": analysis_ds,
            "dbs_analysis": dbs_analysis_ds,
        }
    )
    dt.attrs = {
        "title": output_filename_suffix,
        "software_name": "scotty-beam-tracing",
        "software_version": __version__,
        "date_created": str(datetime.datetime.now()),
        # "id": str(run_id),
    }
    dt.to_netcdf(
        filepath=output_path / f"scotty_output{output_filename_suffix}.h5",
        engine="h5netcdf",
        invalid_netcdf=True,
    )

    if return_dt_field:
        return dt, field
    else:
        return dt


def make_fit(
    method: Optional[Union[str, ProfileFitLike]],
    poloidal_flux_zero_density: float,
    parameters: Optional[Sequence],
    filename: Optional[PathLike],
    quantity: str = "density",
) -> ProfileFitLike:
    """Profile fit for the electron density or temperature (``quantity``).
    ``poloidal_flux_zero_density`` is the flux beyond which the profile is
    zero, for whichever quantity it is"""

    log = logging.getLogger(__name__)
    log.debug(f"Making {quantity} fit profile")

    if callable(method):
        return method

    if not isinstance(method, (str, type(None))):
        raise TypeError(
            f"Unexpected method type. Expected callable, str, or None, but got '{type(method)}'"
        )

    if parameters is None:
        raise ValueError(
            f"Passing `{quantity}_fit_method` ({method}) as string or None requires a list or array of parameters"
        )

    return profile_fit(method, poloidal_flux_zero_density, parameters, filename)
