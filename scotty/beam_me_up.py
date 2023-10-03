# -*- coding: utf-8 -*-
# Copyright 2018 - 2023, Valerian Hall-Chen and the Scotty contributors
# SPDX-License-Identifier: GPL-3.0

"""


Notes
~~~~~

- K**2 = K_R**2 + K_z**2 + (K_zeta/r_R)**2, and K_zeta is constant (mode number). See 14 Sep 2018 notes.
- Start in vacuum, otherwise Psi_3D_beam_initial_cartersian does not get done properly
- Launching from plasma is possible, but beam needs to be initialised differently

Coordinates
~~~~~~~~~~~

- ``X, Y, Z`` - Lab Cartesian coordinates
- ``R, zeta, Z`` - Lab cylindrical coordinates
- ``x, y, g`` - Beam coordinates
- ``u1, u2, u_parallel`` - Field-aligned coordinates

Abbreviations
~~~~~~~~~~~~~

- ``bs`` - backscattered
- ``loc`` - localisation
- ``cum_loc`` - cumulative_localisation
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
- Temperature in KeV
- electron cyclotron frequency positive
- K normalised such that K = 1 in vacuum. (Not implemented yet)
- Distance not normalised yet, should give it thought

"""

from __future__ import annotations
import numpy as np
from scipy import integrate as integrate
import time
import json
import pathlib
import xarray as xr
import datatree
import datetime
import uuid

from scotty.analysis import immediate_analysis, further_analysis
from scotty.fun_general import (
    find_nearest,
    freq_GHz_to_angular_frequency,
    find_Psi_3D_lab,
)
from scotty.fun_evolution import (
    beam_evolution_fun,
    pack_beam_parameters,
    unpack_beam_parameters,
)

# For find_B if using efit files directly
from scotty.profile_fit import profile_fit, ProfileFitLike
from scotty.geometry import (
    MagneticField,
    CircularCrossSectionField,
    ConstantCurrentDensityField,
    InterpolatedField,
    CurvySlabField,
    EFITField,
)
from scotty.hamiltonian import Hamiltonian
from scotty.launch import launch_beam, find_entry_point
from scotty.torbeam import Torbeam
from scotty.ray_solver import propagate_ray
from scotty.plotting import plot_dispersion_relation, plot_poloidal_beam_path
from scotty._version import __version__

# Checks
from scotty.check_input import check_input

# Type hints
from typing import Optional, Union, Sequence, cast
from scotty.typing import PathLike, FloatArray


def beam_me_up(
    poloidal_launch_angle_Torbeam: float,
    toroidal_launch_angle_Torbeam: float,
    launch_freq_GHz: float,
    mode_flag: int,
    launch_beam_width: float,
    launch_beam_curvature: float,
    launch_position: FloatArray,
    # keyword arguments begin
    vacuumLaunch_flag: bool = True,
    relativistic_flag: bool = False,  # includes relativistic corrections to electron mass when set to True
    find_B_method: Union[str, MagneticField] = "torbeam",
    density_fit_parameters: Optional[Sequence] = None,
    temperature_fit_parameters: Optional[Sequence] = None,
    shot=None,
    equil_time=None,
    vacuum_propagation_flag: bool = False,
    Psi_BC_flag: Union[bool, str, None] = None,
    poloidal_flux_enter: float = 1.0,
    poloidal_flux_zero_density: float = 1.0,  ## When polflux >= poloidal_flux_zero_density, Scotty sets density = 0
    poloidal_flux_zero_temperature: float = 1.0,  ## temperature analogue of poloidal_flux_zero_density
    # Finite-difference and solver parameters
    auto_delta_sign=True,  # For flipping signs to maintain forward difference. Applies to delta_R and delta_Z
    delta_R: float = -0.0001,  # in the same units as data_R_coord
    delta_Z: float = 0.0001,  # in the same units as data_Z_coord
    delta_K_R: float = 0.1,  # in the same units as K_R
    delta_K_zeta: float = 0.1,  # in the same units as K_zeta
    delta_K_Z: float = 0.1,  # in the same units as K_z
    interp_order=5,  # For the 2D interpolation functions
    len_tau: int = 102,
    rtol: float = 1e-3,  # for solve_ivp of the beam solver
    atol: float = 1e-6,  # for solve_ivp of the beam solver
    interp_smoothing=0,  # For the 2D interpolation functions. For no smoothing, set to 0
    # Input and output settings
    ne_data_path=pathlib.Path("."),
    magnetic_data_path=pathlib.Path("."),
    Te_data_path=pathlib.Path("."),
    output_path=pathlib.Path("."),
    input_filename_suffix="",
    output_filename_suffix="",
    figure_flag=True,
    detailed_analysis_flag=True,
    # For quick runs (only ray tracing)
    quick_run: bool = False,
    # For launching within the plasma
    plasmaLaunch_K=np.zeros(3),
    plasmaLaunch_Psi_3D_lab_Cartesian=np.zeros([3, 3]),
    density_fit_method: Optional[Union[str, ProfileFitLike]] = None,
    temperature_fit_method: Optional[Union[str, ProfileFitLike]] = None,
    # For circular flux surfaces
    B_T_axis=None,
    B_p_a=None,
    R_axis=None,
    minor_radius_a=None,
) -> datatree.DataTree:
    r"""Run the beam tracer

    Overview
    ========

    1. Initialise density fit parameters. One of:
        - spline with data from file
        - Stefanikova
        - O(3) polynomial
        - tanh
        - quadratic
       See `profile_fit` for more details
    2. If relativistic_flag enabled, initialise temperature
       fit parameters. Not yet fully implemented. One of:
        - spline with data from file
        - linear
        See 'profile_fit' for more details
    3. Initialise magnetic field method. One of:
        - TORBEAM
        - OMFIT
        - analytical
        - EFIT++
        - UDA
        - curvy slab
        - test/test_notime
       See `geometry` for more details
    4. Initialise beam launch parameters (vacuum/plasma). See `launch`
       for more details.
    5. Propagate single ray with IVP solver to find point where beam
       leaves plasma. See `ray_solver` for more details
    6. Propagate beam with IVP solver
    7. Dump raw output
    8. Analysis

    Parameters
    ==========
    poloidal_launch_angle_Torbeam: float
        Poloidal angle of antenna in TORBEAM convention
    toroidal_launch_angle_Torbeam: float
        Toroidal angle of antenna in TORBEAM convention
    launch_freq_GHz: float
        Frequency of the launched beam in GHz
    mode_flag: int
        Either ``+/-1``, used to determine which mode branch to use
    launch_beam_width: float
        Width of the beam at launch
    launch_beam_curvature: float
        Curvatuve of the beam at launch
    launch_position: FloatArray
        Position of the antenna in cylindrical coordinates
    vacuumLaunch_flag: bool
        If ``True``, launch beam from vacuum, otherwise beam launch
        position is inside the plasma already
    vacuum_propagation_flag: bool
        If ``True``, run solver from the launch position, and don't
        use analytical vacuum propagation
    relativistic_flag: bool
        If ``True``, generates a temperature profile from given data
        or parameters and applies relativistic corrections to electron
        mass.
    Psi_BC_flag: String
        If ``None``, do no special treatment at plasma-vacuum boundary
        If ``continuous``, apply BCs for continuous ne but discontinuous gradient of ne
        If ``discontinuous``, apply BCs for discontinuous ne
        Using ``True`` or ``False`` is now deprecated
    poloidal_flux_enter: float
        Normalised poloidal flux label of plasma boundary.
        If vacuum_propagation_flag, then this is where the solver begins.
        If Psi_BC_flag, then this is where the plasma-vacuum BCs are applied
    poloidal_flux_zero_density: float
        At and above this normalised poloidal flux label, Scotty sets the electron density to zero
    poloidal_flux_zero_temperature: float
        At and above this normalised poloidal flux label, Scotty sets the electron temperature to zero.
        This effectively negates any relativistic mass corrections.
    plasmaLaunch_Psi_3D_lab_Cartesian: FloatArray
        :math:`\Psi` of beam in lab Cartesian coordinates. Required if
        ``vacuumLaunch_flag`` is ``False``
    plasmaLaunch_K: FloatArray
        Wavevector of beam at launch. Required if
        ``vacuumLaunch_flag`` is ``False``
    delta_R: float
        Finite difference spacing to use for ``R``
    delta_Z: float
        Finite difference spacing to use for ``Z``
    delta_K_R: float
        Finite difference spacing to use for ``K_R``
    delta_K_zeta: float
        Finite difference spacing to use for ``K_zeta``
    delta_K_Z: float
        Finite difference spacing to use for ``K_Z``
    find_B_method:
        See `create_magnetic_geometry` for more information.

        Common options:

        - ``"efitpp"`` uses magnetic field data from efitpp files
          directly
        - ``"torbeam"`` uses magnetic field data from TORBEAM input
          files
        - ``"omfit"`` is similar to ``"torbeam"`` but reads the data
          from JSON files
        - ``"UDA_saved"`` reads EFIT data from numpy's ``.npz`` files
        - ``"analytical"`` uses a constant current density circular
          equilibrium

        Or pass a `MagneticField` instance directly.

        .. todo:: document which options different methods take

    density_fit_parameters:
        A list of parameters to be passed to the
        ``density_fit_method`` constructor. See the docs for the
        individual methods for the meaning of their parameters

        .. note:: These parameters should *not* include
           ``poloidal_flux_zero_density``
    density_fit_method:
        Parameterisation of the density profile. Either a callable
        (see `ProfileFit`), or one of the following options:

        - ``"smoothing-spline"``: 1D smoothing spline
          (`SmoothingSplineFit`)
        - ``"smoothing-spline-file"``: 1D smoothing spline constructed
          from file (`SmoothingSplineFit.from_dat_file`)
        - ``"stefanikova"``: combination of modified hyperbolic
          :math:`\tan` and a Gaussian (`StefanikovaFit`)
        - ``"poly3"`` or ``"polynomial"``: :math:`n`-th order
          polynomial (`PolynomialFit`)
        - ``"tanh"``: hyperbolic :math:`\tan` (`TanhFit`)
        - ``"quadratic"``: constrained quadratic (`QuadraticFit`)

        If ``density_fit_method`` is a string, then the corresponding
        `DensityFit` object is constructed using
        ``poloidal_flux_zero_density`` and ``density_fit_parameters``.

        ``"smoothing-spline-file"`` looks for a file called
        ``ne<input_filename_suffix>.dat`` in ``ne_data_path``. It also
        uses ``interp_order`` and ``interp_smoothing`` instead of
        ``density_fit_parameters``.

        .. deprecated:: 2.4.0
           If ``None`` (the default) is passed, the method will be
           guessed from the length of ``density_fit_parameters``. In
           this case, ``quadratic`` and ``tanh`` parameters _should_
           include ``poloidal_flux_zero_density`` as the last value.
    temperature_fit_method:
        Parameterisation of the temperature profile. Leverages the
        same ProfileFit class used for density. See density_fit_method
        for details.
    len_tau: int
        Number of output ``tau`` points
    rtol: float
        Relative tolerance for ODE solver
    atol: float
        Absolute tolerance for ODE solver
    quick_run: bool
        If true, then run only the ray tracer and get an analytic
        estimate of the :math:`K` cut-off location
    auto_delta_sign:
        Boolean. Ensures that forward
        difference is always in negative poloidal flux gradient
        direction (into the plasma).
    """

    print("Beam trace me up, Scotty!")
    print(f"scotty version {__version__}")
    run_id = uuid.uuid4()
    print(f"Run ID: {run_id}")

    # ------------------------------
    # Input data #
    # ------------------------------

    # Tidying up the input data
    launch_angular_frequency = freq_GHz_to_angular_frequency(launch_freq_GHz)

    # Ensure paths are `pathlib.Path`
    ne_data_path = pathlib.Path(ne_data_path)
    magnetic_data_path = pathlib.Path(magnetic_data_path)
    Te_data_path = pathlib.Path(Te_data_path)
    output_path = pathlib.Path(output_path)

    # Experimental Profile----------

    # So that saving the input data later does not complain
    ne_data_density_array = None
    ne_data_radialcoord_array = None

    if density_fit_parameters is None and (
        density_fit_method in [None, "smoothing-spline-file"]
    ):
        ne_filename = ne_data_path / f"ne{input_filename_suffix}.dat"
        density_fit_parameters = [ne_filename, interp_order, interp_smoothing]

        # FIXME: Read data so it can be saved later
        ne_data = np.fromfile(ne_filename, dtype=float, sep="   ")
        ne_data_density_array = ne_data[2::2]
        ne_data_radialcoord_array = ne_data[1::2]
    else:
        ne_filename = None

    find_density_1D = make_density_fit(
        density_fit_method,
        poloidal_flux_zero_density,
        density_fit_parameters,
        ne_filename,
    )

    # Run section if relativistic flag is set to True, and checks for whether the required temperature
    # data files are stored in path. If it does not exist, set relativistic flag to false and print
    # a warning, but run the rest of the code.

    Te_filename = None
    if relativistic_flag:
        if temperature_fit_parameters is None and (
            temperature_fit_method
            in [
                None,
                "smoothing-spline-file",
            ]
        ):
            Te_filename = Te_data_path / f"Te{input_filename_suffix}.dat"
            temperature_fit_parameters = [Te_filename, interp_order, interp_smoothing]

            # Modify to test for whether files exist in path, and assign None value to Te_filename if no such files exist.

            # FIXME: Read data so it can be saved later
            # Te_data = np.fromfile(Te_filename, dtype=float, sep="   ")
            # Te_data_density_array = Te_data[2::2]
            # Te_data_radialcoord_array = Te_data[1::2]

    if relativistic_flag:
        find_temperature_1D = make_temperature_fit(
            temperature_fit_method,
            poloidal_flux_zero_temperature,
            temperature_fit_parameters,
            Te_filename,
        )
    else:
        find_temperature_1D = None

    field = create_magnetic_geometry(
        find_B_method,
        magnetic_data_path,
        input_filename_suffix,
        interp_order,
        interp_smoothing,
        B_T_axis,
        R_axis,
        minor_radius_a,
        B_p_a,
        shot,
        equil_time,
        abs(delta_R),
        abs(delta_Z),
    )

    if auto_delta_sign:
        # Flips the sign of delta_Z depending on the orientation of the poloidal flux surface at the point which the ray enters the plasma.
        # This is to ensure a forward difference across the plasma boundary. We expect poloidal flux to decrease in the direction of the plasma.

        entry_coords = find_entry_point(
            launch_position,
            np.deg2rad(poloidal_launch_angle_Torbeam),
            np.deg2rad(toroidal_launch_angle_Torbeam),
            poloidal_flux_enter,
            field,
        )
        entry_R, entry_zeta, entry_Z = entry_coords

        d_poloidal_flux_dR = field.d_poloidal_flux_dR(entry_R, entry_Z, delta_R)
        d_poloidal_flux_dZ = field.d_poloidal_flux_dZ(entry_R, entry_Z, delta_Z)
        # print("Gradients at entry point for Z: ", Z_gradient, ", R: ", R_gradient)

        if d_poloidal_flux_dZ > 0:
            delta_Z = -1 * abs(delta_Z)
        else:
            delta_Z = abs(delta_Z)
        if d_poloidal_flux_dR > 0:
            delta_R = -1 * abs(delta_R)
        else:
            delta_R = abs(delta_R)

    # Initialises H
    hamiltonian = Hamiltonian(
        field,
        launch_angular_frequency,
        mode_flag,
        find_density_1D,
        delta_R,
        delta_Z,
        delta_K_R,
        delta_K_zeta,
        delta_K_Z,
        find_temperature_1D,
    )

    # Checking input data
    check_input(
        mode_flag,
        poloidal_flux_enter,
        launch_position,
        field,
        poloidal_flux_zero_density,
    )

    # -------------------
    # Launch parameters
    # -------------------
    if vacuumLaunch_flag:
        print("Beam launched from outside the plasma")
        (
            K_initial,
            initial_position,
            launch_K,
            Psi_3D_lab_initial,
            Psi_3D_lab_launch,
            Psi_3D_lab_entry,
            Psi_3D_lab_entry_cartersian,
            distance_from_launch_to_entry,
        ) = launch_beam(
            toroidal_launch_angle_Torbeam=toroidal_launch_angle_Torbeam,
            poloidal_launch_angle_Torbeam=poloidal_launch_angle_Torbeam,
            launch_beam_width=launch_beam_width,
            launch_beam_curvature=launch_beam_curvature,
            launch_position=launch_position,
            launch_angular_frequency=launch_angular_frequency,
            mode_flag=mode_flag,
            field=field,
            hamiltonian=hamiltonian,
            vacuum_propagation_flag=vacuum_propagation_flag,
            Psi_BC_flag=Psi_BC_flag,
            poloidal_flux_enter=poloidal_flux_enter,
            delta_R=delta_R,
            delta_Z=delta_Z,
        )
    else:
        print("Beam launched from inside the plasma")
        Psi_3D_lab_initial = find_Psi_3D_lab(
            plasmaLaunch_Psi_3D_lab_Cartesian,
            launch_position[0],
            launch_position[1],
            plasmaLaunch_K[0],
            plasmaLaunch_K[1],
        )
        K_initial = plasmaLaunch_K
        initial_position = launch_position
        launch_K = None
        Psi_3D_lab_launch = None
        Psi_3D_lab_entry = None
        Psi_3D_lab_entry_cartersian = None
        distance_from_launch_to_entry = None

    K_R_initial, K_zeta_initial, K_Z_initial = K_initial

    # -------------------
    # Propagate the ray

    print("Starting the solvers")
    ray_solver_output = propagate_ray(
        poloidal_flux_enter,
        launch_angular_frequency,
        field,
        initial_position,
        K_initial,
        hamiltonian,
        rtol,
        atol,
        quick_run,
        len_tau,
    )
    if quick_run:
        return ray_solver_output

    tau_leave, tau_points = cast(tuple, ray_solver_output)

    # -------------------
    # Propagate the beam

    # Initial conditions for the solver
    beam_parameters_initial = pack_beam_parameters(
        initial_position[0],
        initial_position[1],
        initial_position[2],
        K_R_initial,
        K_Z_initial,
        Psi_3D_lab_initial,
    )

    solver_start_time = time.time()

    solver_beam_output = integrate.solve_ivp(
        beam_evolution_fun,
        [0, tau_leave],
        beam_parameters_initial,
        method="RK45",
        t_eval=tau_points,
        dense_output=False,
        events=None,
        vectorized=False,
        args=(K_zeta_initial, hamiltonian),
        rtol=rtol,
        atol=atol,
    )

    solver_end_time = time.time()
    solver_time = solver_end_time - solver_start_time
    print(f"Time taken (beam solver) {solver_time}s")
    print(f"Number of beam evolution evaluations: {solver_beam_output.nfev}")
    print(
        f"Time per beam evolution evaluation: {solver_time / solver_beam_output.nfev}"
    )

    beam_parameters = solver_beam_output.y
    tau_array = solver_beam_output.t
    solver_status = solver_beam_output.status

    (
        q_R_array,
        q_zeta_array,
        q_Z_array,
        K_R_array,
        K_Z_array,
        Psi_3D_output,
    ) = unpack_beam_parameters(beam_parameters)

    print("Main loop complete")
    # -------------------

    inputs = xr.Dataset(
        {
            "B_T_axis": B_T_axis,
            "B_p_a": B_p_a,
            "K_initial": (["col"], K_initial),
            "Psi_BC_flag": Psi_BC_flag,
            "R_axis": R_axis,
            "atol": atol,
            "delta_K_R": delta_K_R,
            "delta_K_Z": delta_K_Z,
            "delta_K_zeta": delta_K_zeta,
            "delta_R": delta_R,
            "delta_Z": delta_Z,
            "density_fit_method": str(density_fit_method),
            "density_fit_parameters": str(density_fit_parameters),
            "detailed_analysis_flag": detailed_analysis_flag,
            "equil_time": (equil_time),
            "figure_flag": figure_flag,
            "find_B_method": str(find_B_method),
            "initial_position": (["col"], initial_position),
            "input_filename_suffix": input_filename_suffix,
            "interp_order": interp_order,
            "interp_smoothing": interp_smoothing,
            "launch_K": (launch_K),
            "launch_angular_frequency": launch_angular_frequency,
            "launch_beam_curvature": launch_beam_curvature,
            "launch_beam_width": launch_beam_width,
            "launch_freq_GHz": launch_freq_GHz,
            "launch_position": (["col"], launch_position),
            "len_tau": len_tau,
            "magnetic_data_path": str(magnetic_data_path),
            "minor_radius_a": minor_radius_a,
            "mode_flag": mode_flag,
            "ne_data_density_array": (ne_data_density_array),
            "ne_data_path": str(ne_data_path),
            "ne_data_radialcoord_array": (ne_data_radialcoord_array),
            "output_filename_suffix": output_filename_suffix,
            "output_path": str(output_path),
            "plasmaLaunch_K": plasmaLaunch_K,
            "plasmaLaunch_Psi_3D_lab_Cartesian": (
                ["row", "col"],
                plasmaLaunch_Psi_3D_lab_Cartesian,
            ),
            "poloidalFlux_grid": (["R", "Z"], field.poloidalFlux_grid),
            "poloidal_flux_enter": poloidal_flux_enter,
            "poloidal_launch_angle_Torbeam": poloidal_launch_angle_Torbeam,
            "Psi_3D_lab_initial": (
                ["row", "col"],
                Psi_3D_lab_initial,
            ),
            "quick_run": quick_run,
            "rtol": rtol,
            "shot": shot,
            "toroidal_launch_angle_Torbeam": toroidal_launch_angle_Torbeam,
            "vacuumLaunch_flag": vacuumLaunch_flag,
            "vacuum_propagation_flag": vacuum_propagation_flag,
        },
        coords={
            "R": field.R_coord,
            "Z": field.Z_coord,
            "row": ["R", "zeta", "Z"],
            "col": ["R", "zeta", "Z"],
        },
    )
    solver_output = xr.Dataset(
        {
            "solver_status": solver_status,
            "q_R": (["tau"], q_R_array, {"long_name": "R", "units": "m"}),
            "q_zeta": (["tau"], q_zeta_array, {"long_name": r"$\zeta$", "units": "m"}),
            "q_Z": (["tau"], q_Z_array, {"long_name": "Z", "units": "m"}),
            "K_R": (["tau"], K_R_array),
            "K_Z": (["tau"], K_Z_array),
            "Psi_3D": (["tau", "row", "col"], Psi_3D_output),
        },
        coords={
            "tau": tau_array,
            "row": ["R", "zeta", "Z"],
            "col": ["R", "zeta", "Z"],
        },
    )

    dt = datatree.DataTree.from_dict({"inputs": inputs, "solver_output": solver_output})
    dt.attrs = {
        "title": output_filename_suffix,
        "software_name": "scotty-beam-tracing",
        "software_version": __version__,
        "date_created": str(datetime.datetime.now()),
        "id": str(run_id),
    }

    if solver_status == -1:
        # If the solver doesn't finish, end the function here
        print("Solver did not reach completion")
        return

    # -------------------
    # Process the data from the main loop to give a bunch of useful stuff
    # -------------------
    print("Analysing data")
    dH = hamiltonian.derivatives(
        q_R_array, q_Z_array, K_R_array, K_zeta_initial, K_Z_array, second_order=True
    )

    df = immediate_analysis(
        solver_output,
        field,
        find_density_1D,
        find_temperature_1D,
        hamiltonian,
        K_zeta_initial,
        launch_angular_frequency,
        mode_flag,
        delta_R,
        delta_Z,
        delta_K_R,
        delta_K_zeta,
        delta_K_Z,
        Psi_3D_lab_launch,
        Psi_3D_lab_entry,
        distance_from_launch_to_entry,
        vacuumLaunch_flag,
        output_path,
        output_filename_suffix,
        dH,
    )
    analysis = further_analysis(
        inputs,
        df,
        Psi_3D_lab_entry_cartersian,
        output_path,
        output_filename_suffix,
        field,
        detailed_analysis_flag,
        dH,
    )
    df.update(analysis)
    dt["analysis"] = datatree.DataTree(df)

    # We need to use h5netcdf and invalid_netcdf in order to easily
    # write complex numbers
    dt.to_netcdf(
        output_path / f"scotty_output{output_filename_suffix}.h5",
        engine="h5netcdf",
        invalid_netcdf=True,
    )

    if figure_flag:
        default_plots(dt, field, output_path, output_filename_suffix)

    return dt


def default_plots(
    dt: datatree.DataTree, field: MagneticField, output_path: pathlib.Path, suffix: str
) -> None:
    """Save some simple figures

    Allows one to quickly gain an insight into what transpired in the simulation
    """

    print("Making figures")
    plot_poloidal_beam_path(dt, filename=(output_path / f"Ray1_{suffix}.png"))
    plot_dispersion_relation(dt.analysis, filename=(output_path / f"H_{suffix}.png"))
    print("Figures have been saved")


def make_density_fit(
    method: Optional[Union[str, ProfileFitLike]],
    poloidal_flux_zero_density: float,
    parameters: Optional[Sequence],
    filename: Optional[PathLike],
) -> ProfileFitLike:
    """Either construct a `ProfileFit` instance, or return ``method``
    if it's already suitable. Suitable methods are callables that take
    an array of poloidal fluxes and return an array of densities.

    """
    if callable(method):
        return method

    if not isinstance(method, (str, type(None))):
        raise TypeError(
            f"Unexpected method type, expected callable, str or None, got '{type(method)}'"
        )

    if parameters is None:
        raise ValueError(
            f"Passing `density_fit_method` ({method}) as string or None requires a list/array of parameters"
        )

    return profile_fit(method, poloidal_flux_zero_density, parameters, filename)


def make_temperature_fit(
    method: Optional[Union[str, ProfileFitLike]],
    poloidal_flux_zero_temperature: float,
    parameters: Optional[Sequence],
    filename: Optional[PathLike],
) -> (
    ProfileFitLike
):  # Temporary measure to check if DensityFit is compatible with temp data
    """Either construct a `DensityFit` instance, or return ``method``
    if it's already suitable. Suitable methods are callables that take
    an array of poloidal fluxes and return an array of temperatures.

    """
    if callable(method):
        return method

    if not isinstance(method, (str, type(None))):
        raise TypeError(
            f"Unexpected method type, expected callable, str or None, got '{type(method)}'"
        )

    if parameters is None:
        raise ValueError(
            f"Passing `temperature_fit_method` ({method}) as string or None requires a list/array of parameters"
        )

    return profile_fit(method, poloidal_flux_zero_temperature, parameters, filename)


def create_magnetic_geometry(
    find_B_method: Union[str, MagneticField],
    magnetic_data_path: Optional[pathlib.Path] = None,
    input_filename_suffix: str = "",
    interp_order: int = 5,
    interp_smoothing: int = 0,
    B_T_axis: Optional[float] = None,
    R_axis: Optional[float] = None,
    minor_radius_a: Optional[float] = None,
    B_p_a: Optional[float] = None,
    shot: Optional[int] = None,
    equil_time: Optional[float] = None,
    delta_R: Optional[float] = None,
    delta_Z: Optional[float] = None,
    **kwargs,
) -> MagneticField:
    """Create an object representing the magnetic field geometry"""

    if isinstance(find_B_method, MagneticField):
        return find_B_method

    def missing_arg(argument: str) -> str:
        return f"Missing '{argument}' for find_B_method='{find_B_method}'"

    # Analytical geometries

    if find_B_method == "analytical":
        print("Analytical constant current density geometry")
        if B_T_axis is None:
            raise ValueError(missing_arg("B_T_axis"))
        if R_axis is None:
            raise ValueError(missing_arg("R_axis"))
        if minor_radius_a is None:
            raise ValueError(missing_arg("minor_radius_a"))
        if B_p_a is None:
            raise ValueError(missing_arg("B_p_a"))

        return ConstantCurrentDensityField(B_T_axis, R_axis, minor_radius_a, B_p_a)

    if find_B_method == "curvy_slab":
        print("Analytical curvy slab geometry")
        if B_T_axis is None:
            raise ValueError(missing_arg("B_T_axis"))
        if R_axis is None:
            raise ValueError(missing_arg("R_axis"))
        return CurvySlabField(B_T_axis, R_axis)

    if find_B_method == "unit-tests":
        print("Analytical circular cross-section geometry")
        if B_T_axis is None:
            raise ValueError(missing_arg("B_T_axis"))
        if R_axis is None:
            raise ValueError(missing_arg("R_axis"))
        if minor_radius_a is None:
            raise ValueError(missing_arg("minor_radius_a"))
        if B_p_a is None:
            raise ValueError(missing_arg("B_p_a"))

        return CircularCrossSectionField(B_T_axis, R_axis, minor_radius_a, B_p_a)

    ########################################
    # Interpolated numerical geometries from file

    if magnetic_data_path is None:
        raise ValueError(missing_arg("magnetic_data_path"))

    if find_B_method == "torbeam":
        print("Using Torbeam input files for B and poloidal flux")

        # topfile
        # Others: inbeam.dat, Te.dat (not currently used in this code)
        topfile_filename = magnetic_data_path / f"topfile{input_filename_suffix}"
        torbeam = Torbeam.from_file(topfile_filename)

        return InterpolatedField(
            torbeam.R_grid,
            torbeam.Z_grid,
            torbeam.B_R,
            torbeam.B_T,
            torbeam.B_Z,
            torbeam.psi,
            interp_order,
            interp_smoothing,
        )

    elif find_B_method == "omfit":
        print("Using OMFIT JSON Torbeam file for B and poloidal flux")
        topfile_filename = magnetic_data_path / f"topfile{input_filename_suffix}.json"

        with open(topfile_filename) as f:
            data = json.load(f)

        data_R_coord = np.array(data["R"])
        data_Z_coord = np.array(data["Z"])

        def unflatten(array):
            """Convert from column-major (TORBEAM, Fortran) to row-major order (Scotty, Python)"""
            return np.asarray(array).reshape(len(data_Z_coord), len(data_R_coord)).T

        return InterpolatedField(
            data_R_coord,
            data_Z_coord,
            unflatten(data["Br"]),
            unflatten(data["Bt"]),
            unflatten(data["Bz"]),
            unflatten(data["pol_flux"]),
            interp_order,
            interp_smoothing,
        )

    if find_B_method == "test":
        # Works nicely with the new MAST-U UDA output
        filename = magnetic_data_path / f"{shot}_equilibrium_data.npz"
        with np.load(filename) as loadfile:
            time_EFIT = loadfile["time_EFIT"]
            t_idx = find_nearest(time_EFIT, equil_time)
            print("EFIT time", time_EFIT[t_idx])

            return InterpolatedField(
                R_grid=loadfile["R_EFIT"],
                Z_grid=loadfile["Z_EFIT"],
                psi=loadfile["poloidalFlux_grid"][t_idx, :, :],
                B_T=loadfile["Bphi_grid"][t_idx, :, :],
                B_R=loadfile["Br_grid"][t_idx, :, :],
                B_Z=loadfile["Bz_grid"][t_idx, :, :],
                interp_order=interp_order,
                interp_smoothing=interp_smoothing,
            )
    if find_B_method == "test_notime":
        with np.load(magnetic_data_path) as loadfile:
            return InterpolatedField(
                R_grid=loadfile["R_EFIT"],
                Z_grid=loadfile["Z_EFIT"],
                psi=loadfile["poloidalFlux_grid"],
                B_T=loadfile["Bphi_grid"],
                B_R=loadfile["Br_grid"],
                B_Z=loadfile["Bz_grid"],
                interp_order=interp_order,
                interp_smoothing=interp_smoothing,
            )

    ########################################
    # Interpolated numerical geometries from EFIT data

    if find_B_method == "UDA_saved" and shot is None:
        print("Using MAST(-U) saved UDA data")
        # If using this part of the code, magnetic_data_path needs to include the filename
        # Assuming only one time present in file
        with np.load(magnetic_data_path) as loadfile:
            return EFITField(
                R_grid=loadfile["R_EFIT"],
                Z_grid=loadfile["Z_EFIT"],
                rBphi=loadfile["rBphi"],
                psi_norm_2D=loadfile["poloidalFlux_grid"],
                psi_unnorm_axis=loadfile["poloidalFlux_unnormalised_axis"],
                psi_unnorm_boundary=loadfile["poloidalFlux_unnormalised_boundary"],
                delta_R=delta_R,
                delta_Z=delta_Z,
                interp_order=interp_order,
                interp_smoothing=interp_smoothing,
            )

    # Data files with multiple time-slices

    if equil_time is None:
        raise ValueError(missing_arg("equil_time"))

    if find_B_method == "EFITpp":
        print(
            "Using MSE-constrained EFIT++ output files directly for B and poloidal flux"
        )
        return EFITField.from_EFITpp(
            magnetic_data_path / "efitOut.nc",
            equil_time,
            delta_R,
            delta_Z,
            interp_order,
            interp_smoothing,
        )

    if find_B_method == "UDA_saved" and shot is not None and shot <= 30471:  # MAST
        print(f"Using MAST shot {shot} from saved UDA data")
        # 30471 is the last shot on MAST
        # data saved differently for MAST-U shots
        filename = magnetic_data_path / f"{shot}_equilibrium_data.npz"
        return EFITField.from_MAST_saved(
            filename, equil_time, delta_R, delta_Z, interp_order, interp_smoothing
        )

    if find_B_method == "UDA_saved" and shot is not None and shot > 30471:  # MAST-U
        print(f"Using MAST-U shot {shot} from saved UDA data")
        # 30471 is the last shot on MAST
        # data saved differently for MAST-U shots
        filename = magnetic_data_path / f"{shot}_equilibrium_data.npz"
        return EFITField.from_MAST_U_saved(
            filename, equil_time, delta_R, delta_Z, interp_order, interp_smoothing
        )

    raise ValueError(f"Invalid find_B_method '{find_B_method}'")
