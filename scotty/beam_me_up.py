# -*- coding: utf-8 -*-
# Copyright 2018 - 2023, Valerian Hall-Chen and the Scotty contributors
# SPDX-License-Identifier: GPL-3.0

"""

Plan
~~~~

Output everything to a file, and then do the analysis on that file.

1) Check that gradK_xi and such are done correctly, check that K_mag is calculated correctly when K_zeta is nonzero
2) Check that the calculation of Psi makes sense (and the rotation angle)
3) Check that K initial's calculation makes sense


Notes
~~~~~

- The loading of the input files was taken from integral_5 and modified.
- I should launch the beam inside the last closed flux surface
- K**2 = K_R**2 + K_z**2 + (K_zeta/r_R)**2, and K_zeta is constant (mode number). See 14 Sep 2018 notes.

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
- Start in vacuum, otherwise Psi_3D_beam_initial_cartersian does not get done properly

"""

from __future__ import annotations
import numpy as np
from scipy import integrate as integrate
from scipy import constants as constants
import matplotlib.pyplot as plt
import time
import json
import pathlib

from scotty.fun_general import (
    find_nearest,
    contract_special,
    make_unit_vector_from_cross_product,
    find_x0,
    find_waist,
    freq_GHz_to_angular_frequency,
    angular_frequency_to_wavenumber,
    find_Psi_3D_lab,
)
from scotty.fun_general import find_q_lab_Cartesian, find_Psi_3D_lab_Cartesian
from scotty.fun_general import (
    find_normalised_plasma_freq,
    find_normalised_gyro_freq,
    find_electron_mass,
)
from scotty.fun_general import find_epsilon_para, find_epsilon_perp, find_epsilon_g
from scotty.fun_general import find_dbhat_dR, find_dbhat_dZ
from scotty.fun_general import find_H_Cardano, find_D
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
from scotty.hamiltonian import Hamiltonian, hessians
from scotty.launch import launch_beam, find_entry_point
from scotty.torbeam import Torbeam
from scotty.ray_solver import propagate_ray
from scotty._version import __version__

# Checks
from scotty.check_input import check_input
from scotty.check_output import check_output

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
    verbose_output_flag=True,
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
    # For flipping signs to maintain forward difference
    auto_delta_sign_Z=1,
    auto_delta_sign_R=1,
):
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
    auto_delta_sign_Z:
        A sign variable that takes on +1/-1. Ensures that forward
        differebce is always in negative poloidal flux gradient
        direction (into the plasma).
        Current implementation is flawed because the MagneticField
        has to be constructed in order to determine the intersection
        point and corresponding sign. Sign flip is thus subsequently
        applied throughout the code except for the already constructed
        field.
    auto_delta_sign_R:
        Similar to auto_delta_sign_Z but for the R coordinate

    """

    # major_radius = 0.9

    print("Beam trace me up, Scotty!")
    print(f"scotty version {__version__}")
    # ------------------------------

    # ------------------------------
    # Input data #
    # ------------------------------

    # Tidying up the input data
    launch_angular_frequency = freq_GHz_to_angular_frequency(launch_freq_GHz)
    wavenumber_K0 = angular_frequency_to_wavenumber(launch_angular_frequency)

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
        delta_R,
        delta_Z,
    )

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

    Z_gradient = (
        field.poloidal_flux(entry_R, entry_Z + delta_Z)
        - field.poloidal_flux(entry_R, entry_Z - delta_Z)
    ) / abs(delta_Z)
    R_gradient = (
        field.poloidal_flux(entry_R + delta_R, entry_Z)
        - field.poloidal_flux(entry_R - delta_R, entry_Z)
    ) / abs(delta_R)
    print("Gradients at entry point for Z: ", Z_gradient, ", R: ", R_gradient)

    if Z_gradient > 0:
        auto_delta_sign_Z = -1
    if R_gradient > 0:
        auto_delta_sign_R = -1

    # Modify to take in an optional find_temp_1D argument
    hamiltonian = Hamiltonian(
        field,
        launch_angular_frequency,
        mode_flag,
        find_density_1D,
        auto_delta_sign_R * delta_R,
        auto_delta_sign_Z * delta_Z,
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
            delta_R=auto_delta_sign_R * delta_R,
            delta_Z=auto_delta_sign_Z * delta_Z,
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

    numberOfDataPoints = len(tau_array)

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

    # -------------------
    # This saves the data generated by the main loop and the input data
    # -------------------
    if verbose_output_flag:
        print("Saving data")
        np.savez(
            output_path / f"data_input{output_filename_suffix}",
            poloidalFlux_grid=field.poloidalFlux_grid,
            data_R_coord=field.R_coord,
            data_Z_coord=field.Z_coord,
            poloidal_launch_angle_Torbeam=poloidal_launch_angle_Torbeam,
            toroidal_launch_angle_Torbeam=toroidal_launch_angle_Torbeam,
            launch_freq_GHz=launch_freq_GHz,
            mode_flag=mode_flag,
            launch_beam_width=launch_beam_width,
            launch_beam_curvature=launch_beam_curvature,
            launch_position=launch_position,
            launch_K=launch_K,
            ne_data_density_array=ne_data_density_array,
            ne_data_radialcoord_array=ne_data_radialcoord_array,
            equil_time=equil_time,
            delta_R=delta_R,
            delta_Z=delta_Z,
            delta_K_R=delta_K_R,
            delta_K_zeta=delta_K_zeta,
            delta_K_Z=delta_K_Z,
            interp_order=interp_order,
            interp_smoothing=interp_smoothing,
            auto_delta_sign_Z=auto_delta_sign_Z,
            auto_delt_sign_R=auto_delta_sign_R,
        )
        np.savez(
            output_path / f"solver_output{output_filename_suffix}",
            solver_status=solver_status,
            tau_array=tau_array,
            q_R_array=q_R_array,
            q_zeta_array=q_zeta_array,
            q_Z_array=q_Z_array,
            K_R_array=K_R_array,
            K_Z_array=K_Z_array,
            Psi_3D_output=Psi_3D_output,
        )

    if solver_status == -1:
        # If the solver doesn't finish, end the function here
        print("Solver did not reach completion")
        return

    # -------------------
    # Generates additional data along the path of the beam
    # -------------------

    # Calculate various properties along the ray
    poloidal_flux_output = field.poloidal_flux(q_R_array, q_Z_array)
    electron_density_output = np.asfarray(find_density_1D(poloidal_flux_output))
    if relativistic_flag and find_temperature_1D:
        temperature_output = np.asfarray(find_temperature_1D(poloidal_flux_output))
    else:
        temperature_output = None

    dH = hamiltonian.derivatives(
        q_R_array, q_Z_array, K_R_array, K_zeta_initial, K_Z_array, second_order=True
    )

    dH_dR_output = dH["dH_dR"]
    dH_dZ_output = dH["dH_dZ"]
    dH_dKR_output = dH["dH_dKR"]
    dH_dKzeta_output = dH["dH_dKzeta"]
    dH_dKZ_output = dH["dH_dKZ"]

    # Calculates nabla_K H
    # Calculates g_hat
    g_hat_output = np.zeros([numberOfDataPoints, 3])
    g_magnitude_output = (
        q_R_array**2 * dH_dKzeta_output**2 + dH_dKR_output**2 + dH_dKZ_output**2
    ) ** 0.5
    g_hat_output[:, 0] = dH_dKR_output / g_magnitude_output  # g_hat_R
    g_hat_output[:, 1] = q_R_array * dH_dKzeta_output / g_magnitude_output  # g_hat_zeta
    g_hat_output[:, 2] = dH_dKZ_output / g_magnitude_output  # g_hat_Z

    # Calculates b_hat and grad_b_hat
    b_hat_output = np.zeros([numberOfDataPoints, 3])
    B_R_output = field.B_R(q_R_array, q_Z_array)
    B_T_output = field.B_T(q_R_array, q_Z_array)
    B_Z_output = field.B_Z(q_R_array, q_Z_array)
    B_magnitude = np.sqrt(B_R_output**2 + B_T_output**2 + B_Z_output**2)
    b_hat_output[:, 0] = B_R_output / B_magnitude
    b_hat_output[:, 1] = B_T_output / B_magnitude
    b_hat_output[:, 2] = B_Z_output / B_magnitude

    grad_bhat_output = np.zeros([numberOfDataPoints, 3, 3])
    dbhat_dR = find_dbhat_dR(
        q_R_array,
        q_Z_array,
        auto_delta_sign_R * delta_R,
        field.B_R,
        field.B_T,
        field.B_Z,
    )
    dbhat_dZ = find_dbhat_dZ(
        q_R_array,
        q_Z_array,
        auto_delta_sign_Z * delta_Z,
        field.B_R,
        field.B_T,
        field.B_Z,
    )
    # Transpose dbhat_dR so that it has the right shape
    grad_bhat_output[:, 0, :] = dbhat_dR.T
    grad_bhat_output[:, 2, :] = dbhat_dZ.T
    grad_bhat_output[:, 1, 0] = -B_T_output / (B_magnitude * q_R_array)
    grad_bhat_output[:, 1, 1] = B_R_output / (B_magnitude * q_R_array)

    # x_hat and y_hat
    y_hat_output = make_unit_vector_from_cross_product(b_hat_output, g_hat_output)
    x_hat_output = make_unit_vector_from_cross_product(y_hat_output, g_hat_output)

    # Components of the dielectric tensor
    epsilon_para_output = find_epsilon_para(
        electron_density_output, launch_angular_frequency, temperature_output
    )
    epsilon_perp_output = find_epsilon_perp(
        electron_density_output,
        B_magnitude,
        launch_angular_frequency,
        temperature_output,
    )
    epsilon_g_output = find_epsilon_g(
        electron_density_output,
        B_magnitude,
        launch_angular_frequency,
        temperature_output,
    )

    # Plasma and cyclotron frequencies
    normalised_plasma_freqs = find_normalised_plasma_freq(
        electron_density_output, launch_angular_frequency, temperature_output
    )
    normalised_gyro_freqs = find_normalised_gyro_freq(
        B_magnitude, launch_angular_frequency, temperature_output
    )

    # -------------------
    # Not useful for physics or data analysis
    # But good for checking whether things are working properly
    # -------------------
    #
    H_output = hamiltonian(q_R_array, q_Z_array, K_R_array, K_zeta_initial, K_Z_array)
    # Create and immediately evaluate a Hamiltonian with the opposite mode
    H_other = Hamiltonian(
        field,
        launch_angular_frequency,
        -mode_flag,
        find_density_1D,
        auto_delta_sign_R * delta_R,
        auto_delta_sign_Z * delta_Z,
        delta_K_R,
        delta_K_zeta,
        delta_K_Z,
    )(q_R_array, q_Z_array, K_R_array, K_zeta_initial, K_Z_array)

    # Gradients of poloidal flux along the ray
    dpolflux_dR_debugging = field.d_poloidal_flux_dR(
        q_R_array, q_Z_array, auto_delta_sign_R * delta_R
    )
    dpolflux_dZ_debugging = field.d_poloidal_flux_dZ(
        q_R_array, q_Z_array, auto_delta_sign_Z * delta_Z
    )

    # -------------------
    # This saves the data generated by the main loop and the input data
    # Input data saved at this point in case something is changed between loading and the end of the main loop, this allows for comparison
    # The rest of the data is save further down, after the analysis generates them.
    # Just in case the analysis fails to run, at least one can get the data from the main loop
    # -------------------

    # Set **save_kwargs for saving temperature
    save_kwargs = {}
    if temperature_output is not None:
        save_kwargs["temperature_output"] = temperature_output

    if vacuumLaunch_flag:
        np.savez(
            output_path / f"data_output{output_filename_suffix}",
            tau_array=tau_array,
            q_R_array=q_R_array,
            q_zeta_array=q_zeta_array,
            q_Z_array=q_Z_array,
            K_R_array=K_R_array,
            K_zeta_initial=K_zeta_initial,
            K_Z_array=K_Z_array,
            Psi_3D_output=Psi_3D_output,
            Psi_3D_lab_launch=Psi_3D_lab_launch,
            Psi_3D_lab_entry=Psi_3D_lab_entry,
            distance_from_launch_to_entry=distance_from_launch_to_entry,
            g_hat_output=g_hat_output,
            g_magnitude_output=g_magnitude_output,
            B_magnitude=B_magnitude,
            B_R_output=B_R_output,
            B_T_output=B_T_output,
            B_Z_output=B_Z_output,
            y_hat_output=y_hat_output,
            x_hat_output=x_hat_output,
            b_hat_output=b_hat_output,
            grad_bhat_output=grad_bhat_output,
            dH_dKR_output=dH_dKR_output,
            dH_dKzeta_output=dH_dKzeta_output,
            dH_dKZ_output=dH_dKZ_output,
            dH_dR_output=dH_dR_output,
            dH_dZ_output=dH_dZ_output,
            # grad_grad_H_output=grad_grad_H_output,gradK_grad_H_output=gradK_grad_H_output,gradK_gradK_H_output=gradK_gradK_H_output,
            dpolflux_dR_debugging=dpolflux_dR_debugging,
            dpolflux_dZ_debugging=dpolflux_dZ_debugging,
            epsilon_para_output=epsilon_para_output,
            epsilon_perp_output=epsilon_perp_output,
            epsilon_g_output=epsilon_g_output,
            electron_density_output=electron_density_output,
            normalised_plasma_freqs=normalised_plasma_freqs,
            normalised_gyro_freqs=normalised_gyro_freqs,
            H_output=H_output,
            H_other=H_other,
            poloidal_flux_output=poloidal_flux_output,
            **save_kwargs
            # dB_dR_FFD_debugging=dB_dR_FFD_debugging,dB_dZ_FFD_debugging=dB_dZ_FFD_debugging,
            # d2B_dR2_FFD_debugging=d2B_dR2_FFD_debugging,d2B_dZ2_FFD_debugging=d2B_dZ2_FFD_debugging,d2B_dR_dZ_FFD_debugging=d2B_dR_dZ_FFD_debugging,
            # poloidal_flux_debugging_1R=poloidal_flux_debugging_1R,
            # poloidal_flux_debugging_2R=poloidal_flux_debugging_2R,
            # poloidal_flux_debugging_3R=poloidal_flux_debugging_3R,
            # poloidal_flux_debugging_1Z=poloidal_flux_debugging_1Z,
            # poloidal_flux_debugging_2Z=poloidal_flux_debugging_2Z,
            # poloidal_flux_debugging_3Z=poloidal_flux_debugging_3Z,
            # poloidal_flux_debugging_2R_2Z=poloidal_flux_debugging_2R_2Z,
            # electron_density_debugging_1R=electron_density_debugging_1R,
            # electron_density_debugging_2R=electron_density_debugging_2R,
            # electron_density_debugging_3R=electron_density_debugging_3R,
            # electron_density_debugging_1Z=electron_density_debugging_1Z,
            # electron_density_debugging_2Z=electron_density_debugging_2Z,
            # electron_density_debugging_3Z=electron_density_debugging_3Z,
            # electron_density_debugging_2R_2Z=electron_density_debugging_2R_2Z,
            # dpolflux_dR_FFD_debugging=dpolflux_dR_FFD_debugging,
            # dpolflux_dZ_FFD_debugging=dpolflux_dZ_FFD_debugging,
            # d2polflux_dR2_FFD_debugging=d2polflux_dR2_FFD_debugging,
            # d2polflux_dZ2_FFD_debugging=d2polflux_dZ2_FFD_debugging,
        )
    else:
        np.savez(
            output_path / f"data_output{output_filename_suffix}",
            tau_array=tau_array,
            q_R_array=q_R_array,
            q_zeta_array=q_zeta_array,
            q_Z_array=q_Z_array,
            K_R_array=K_R_array,
            K_zeta_initial=K_zeta_initial,
            K_Z_array=K_Z_array,
            Psi_3D_output=Psi_3D_output,
            Psi_3D_lab_launch=Psi_3D_lab_launch,
            g_hat_output=g_hat_output,
            g_magnitude_output=g_magnitude_output,
            B_magnitude=B_magnitude,
            B_R_output=B_R_output,
            B_T_output=B_T_output,
            B_Z_output=B_Z_output,
            x_hat_output=x_hat_output,
            y_hat_output=y_hat_output,
            b_hat_output=b_hat_output,
            grad_bhat_output=grad_bhat_output,
            dH_dKR_output=dH_dKR_output,
            dH_dKzeta_output=dH_dKzeta_output,
            dH_dKZ_output=dH_dKZ_output,
            dH_dR_output=dH_dR_output,
            dH_dZ_output=dH_dZ_output,
            # grad_grad_H_output=grad_grad_H_output,gradK_grad_H_output=gradK_grad_H_output,gradK_gradK_H_output=gradK_gradK_H_output,
            # d_poloidal_flux_dR_output=d_poloidal_flux_dR_output,
            # d_poloidal_flux_dZ_output=d_poloidal_flux_dZ_output,
            # epsilon_para_output=epsilon_para_output,epsilon_perp_output=epsilon_perp_output,epsilon_g_output=epsilon_g_output,
            # electron_density_output=electron_density_output,H_output=H_output
        )

    print("Data saved")
    # -------------------

    # -------------------
    # Process the data from the main loop to give a bunch of useful stuff
    # -------------------
    print("Analysing data")

    # Calculates various useful stuff
    [q_X_array, q_Y_array, _] = find_q_lab_Cartesian(
        [q_R_array, q_zeta_array, q_Z_array]
    )
    point_spacing = (
        (np.diff(q_X_array)) ** 2
        + (np.diff(q_Y_array)) ** 2
        + (np.diff(q_Z_array)) ** 2
    ) ** 0.5
    distance_along_line = np.cumsum(point_spacing)
    distance_along_line = np.append(0, distance_along_line)
    RZ_point_spacing = ((np.diff(q_Z_array)) ** 2 + (np.diff(q_R_array)) ** 2) ** 0.5
    RZ_distance_along_line = np.cumsum(RZ_point_spacing)
    RZ_distance_along_line = np.append(0, RZ_distance_along_line)

    # Calculates the index of the minimum magnitude of K
    # That is, finds when the beam hits the cut-off
    K_magnitude_array = (
        K_R_array**2 + K_zeta_initial**2 / q_R_array**2 + K_Z_array**2
    ) ** (0.5)

    # Index of the cutoff, at the minimum value of K, use this with other arrays
    cutoff_index = find_nearest(abs(K_magnitude_array), 0)

    cyclotron_freq_output = launch_angular_frequency * find_normalised_gyro_freq(
        B_magnitude, launch_angular_frequency
    )

    #         # Calculates when the beam 'enters' and 'leaves' the plasma
    #         # Here, entry and exit refer to crossing poloidal_flux_enter, if specified
    #         # Otherwise, entry and exit refer to crossing the LCFS, poloidal_flux = 1.0
    #     if poloidal_flux_enter is None:
    #         in_out_poloidal_flux = 1.0
    #     else:
    #         in_out_poloidal_flux = poloidal_flux_enter
    #     poloidal_flux_a = poloidal_flux_output[0:cutoff_index]
    #     poloidal_flux_b = poloidal_flux_output[cutoff_index::]
    #     in_index = find_nearest(poloidal_flux_a,in_out_poloidal_flux)
    #     out_index = cutoff_index + find_nearest(poloidal_flux_b,in_out_poloidal_flux)

    # Calcuating the angles theta and theta_m
    sin_theta_m_analysis = np.zeros(numberOfDataPoints)
    sin_theta_m_analysis[:] = (
        b_hat_output[:, 0] * K_R_array[:]
        + b_hat_output[:, 1] * K_zeta_initial / q_R_array[:]
        + b_hat_output[:, 2] * K_Z_array[:]
    ) / (
        K_magnitude_array[:]
    )  # B \cdot K / (abs (B) abs(K))

    # Assumes the mismatch angle is never smaller than -90deg or bigger than 90deg
    theta_m_output = np.sign(sin_theta_m_analysis) * np.arcsin(
        abs(sin_theta_m_analysis)
    )

    kperp1_hat_output = make_unit_vector_from_cross_product(y_hat_output, b_hat_output)
    # The negative sign is there by definition
    sin_theta_analysis = -contract_special(x_hat_output, kperp1_hat_output)
    # sin_theta_analysis = -contract_special(g_hat_output,b_hat_output) # The negative sign is there by definition. Alternative way to get sin_theta
    # Assumes theta is never smaller than -90deg or bigger than 90deg
    theta_output = np.sign(sin_theta_analysis) * np.arcsin(abs(sin_theta_analysis))

    cos_theta_analysis = np.cos(theta_output)
    tan_theta_analysis = np.tan(theta_output)
    # -----

    # Calcuating the corrections to make M from Psi
    # Includes terms small in mismatch

    # The dominant value of kperp1 that is backscattered at every point
    k_perp_1_bs = (
        -2
        * K_magnitude_array
        * np.cos(theta_m_output + theta_output)
        / cos_theta_analysis
    )
    # k_perp_1_bs = -2 * K_magnitude_array # when mismatch is small

    # Converting x_hat, y_hat, and Psi_3D to Cartesians so we can contract them with each other
    y_hat_Cartesian = np.zeros([numberOfDataPoints, 3])
    x_hat_Cartesian = np.zeros([numberOfDataPoints, 3])
    g_hat_Cartesian = np.zeros([numberOfDataPoints, 3])
    y_hat_Cartesian[:, 0] = y_hat_output[:, 0] * np.cos(q_zeta_array) - y_hat_output[
        :, 1
    ] * np.sin(q_zeta_array)
    y_hat_Cartesian[:, 1] = y_hat_output[:, 0] * np.sin(q_zeta_array) + y_hat_output[
        :, 1
    ] * np.cos(q_zeta_array)
    y_hat_Cartesian[:, 2] = y_hat_output[:, 2]
    x_hat_Cartesian[:, 0] = x_hat_output[:, 0] * np.cos(q_zeta_array) - x_hat_output[
        :, 1
    ] * np.sin(q_zeta_array)
    x_hat_Cartesian[:, 1] = x_hat_output[:, 0] * np.sin(q_zeta_array) + x_hat_output[
        :, 1
    ] * np.cos(q_zeta_array)
    x_hat_Cartesian[:, 2] = x_hat_output[:, 2]
    g_hat_Cartesian[:, 0] = g_hat_output[:, 0] * np.cos(q_zeta_array) - g_hat_output[
        :, 1
    ] * np.sin(q_zeta_array)
    g_hat_Cartesian[:, 1] = g_hat_output[:, 0] * np.sin(q_zeta_array) + g_hat_output[
        :, 1
    ] * np.cos(q_zeta_array)
    g_hat_Cartesian[:, 2] = g_hat_output[:, 2]

    Psi_3D_Cartesian = find_Psi_3D_lab_Cartesian(
        Psi_3D_output, q_R_array, q_zeta_array, K_R_array, K_zeta_initial
    )
    Psi_xx_output = contract_special(
        x_hat_Cartesian, contract_special(Psi_3D_Cartesian, x_hat_Cartesian)
    )
    Psi_xy_output = contract_special(
        x_hat_Cartesian, contract_special(Psi_3D_Cartesian, y_hat_Cartesian)
    )
    Psi_yy_output = contract_special(
        y_hat_Cartesian, contract_special(Psi_3D_Cartesian, y_hat_Cartesian)
    )
    Psi_xg_output = contract_special(
        x_hat_Cartesian, contract_special(Psi_3D_Cartesian, g_hat_Cartesian)
    )
    Psi_yg_output = contract_special(
        y_hat_Cartesian, contract_special(Psi_3D_Cartesian, g_hat_Cartesian)
    )
    Psi_gg_output = contract_special(
        g_hat_Cartesian, contract_special(Psi_3D_Cartesian, g_hat_Cartesian)
    )

    Psi_xx_entry = np.dot(
        x_hat_Cartesian[0, :],
        np.dot(Psi_3D_lab_entry_cartersian, x_hat_Cartesian[0, :]),
    )
    Psi_xy_entry = np.dot(
        x_hat_Cartesian[0, :],
        np.dot(Psi_3D_lab_entry_cartersian, y_hat_Cartesian[0, :]),
    )
    Psi_yy_entry = np.dot(
        y_hat_Cartesian[0, :],
        np.dot(Psi_3D_lab_entry_cartersian, y_hat_Cartesian[0, :]),
    )

    # Calculating intermediate terms that are needed for the corrections in M
    xhat_dot_grad_bhat = contract_special(x_hat_output, grad_bhat_output)
    yhat_dot_grad_bhat = contract_special(y_hat_output, grad_bhat_output)
    ray_curvature_kappa_output = np.zeros([numberOfDataPoints, 3])
    ray_curvature_kappa_output[:, 0] = (1 / g_magnitude_output) * (
        np.gradient(g_hat_output[:, 0], tau_array)
        - g_hat_output[:, 1] * dH_dKzeta_output  # See notes 07 June 2021
    )
    ray_curvature_kappa_output[:, 1] = (1 / g_magnitude_output) * (
        np.gradient(g_hat_output[:, 1], tau_array)
        + g_hat_output[:, 0] * dH_dKzeta_output  # See notes 07 June 2021
    )
    ray_curvature_kappa_output[:, 2] = (1 / g_magnitude_output) * np.gradient(
        g_hat_output[:, 2], tau_array
    )
    kappa_magnitude = np.linalg.norm(ray_curvature_kappa_output, axis=-1)
    d_theta_d_tau = np.gradient(theta_output, tau_array)
    d_xhat_d_tau_output = np.zeros([numberOfDataPoints, 3])
    d_xhat_d_tau_output[:, 0] = (
        np.gradient(x_hat_output[:, 0], tau_array)
        - x_hat_output[:, 1] * dH_dKzeta_output
    )  # See notes 07 June 2021
    d_xhat_d_tau_output[:, 1] = (
        np.gradient(x_hat_output[:, 1], tau_array)
        + x_hat_output[:, 0] * dH_dKzeta_output
    )  # See notes 07 June 2021
    d_xhat_d_tau_output[:, 2] = np.gradient(x_hat_output[:, 2], tau_array)

    xhat_dot_grad_bhat_dot_xhat_output = contract_special(
        xhat_dot_grad_bhat, x_hat_output
    )
    xhat_dot_grad_bhat_dot_yhat_output = contract_special(
        xhat_dot_grad_bhat, y_hat_output
    )
    xhat_dot_grad_bhat_dot_ghat_output = contract_special(
        xhat_dot_grad_bhat, g_hat_output
    )
    yhat_dot_grad_bhat_dot_xhat_output = contract_special(
        yhat_dot_grad_bhat, x_hat_output
    )
    yhat_dot_grad_bhat_dot_yhat_output = contract_special(
        yhat_dot_grad_bhat, y_hat_output
    )
    yhat_dot_grad_bhat_dot_ghat_output = contract_special(
        yhat_dot_grad_bhat, g_hat_output
    )
    kappa_dot_xhat_output = contract_special(ray_curvature_kappa_output, x_hat_output)
    kappa_dot_yhat_output = contract_special(ray_curvature_kappa_output, y_hat_output)
    # This should be 0. Good to check.
    kappa_dot_ghat_output = contract_special(ray_curvature_kappa_output, g_hat_output)
    d_xhat_d_tau_dot_yhat_output = contract_special(d_xhat_d_tau_output, y_hat_output)

    # bhat_dot_grad_bhat = contract_special(b_hat_output,grad_bhat_output)
    # bhat_dot_grad_bhat_dot_ghat_output = contract_special(bhat_dot_grad_bhat,g_hat_output)
    # M_xx_output = Psi_xx_output - (k_perp_1_bs/2) * bhat_dot_grad_bhat_dot_ghat_output

    # k_perp_2_bs = 0  # As argued with separation of scales and stuff

    # M_xx_output = (
    #     Psi_xx_output
    #     + (k_perp_1_bs/2) * (
    #         (sin_theta_analysis / g_magnitude_output) * d_theta_d_tau
    #         - kappa_dot_xhat_output * sin_theta_analysis
    #         + xhat_dot_grad_bhat_dot_ghat_output
    #         - xhat_dot_grad_bhat_dot_xhat_output * tan_theta_analysis
    #     )
    #     + (k_perp_2_bs/2) * (
    #         (tan_theta_analysis / g_magnitude_output) *
    #         d_xhat_d_tau_dot_yhat_output
    #         + xhat_dot_grad_bhat_dot_yhat_output / cos_theta_analysis
    #     )
    # )

    # M_xy_output = (
    #     Psi_xy_output
    #     + (k_perp_1_bs/2) * (
    #         - kappa_dot_yhat_output * sin_theta_analysis
    #         + yhat_dot_grad_bhat_dot_ghat_output
    #         + (sin_theta_analysis * tan_theta_analysis /
    #            g_magnitude_output) * d_xhat_d_tau_dot_yhat_output
    #         - yhat_dot_grad_bhat_dot_xhat_output * tan_theta_analysis
    #     )
    #     + (k_perp_2_bs/2) * (
    #         yhat_dot_grad_bhat_dot_yhat_output / cos_theta_analysis
    #     )
    # )

    # M_yy_output = Psi_yy_output

    # Calculates the components of M_w, only taking into consideration
    # correction terms that are not small in mismatch
    M_xx_output = Psi_xx_output + (k_perp_1_bs / 2) * xhat_dot_grad_bhat_dot_ghat_output
    M_xy_output = Psi_xy_output + (k_perp_1_bs / 2) * yhat_dot_grad_bhat_dot_ghat_output
    M_yy_output = Psi_yy_output
    # -----

    # Calculates the localisation, wavenumber resolution, and mismatch attenuation pieces
    det_M_w_analysis = M_xx_output * M_yy_output - M_xy_output**2
    M_w_inv_xx_output = M_yy_output / det_M_w_analysis
    M_w_inv_xy_output = -M_xy_output / det_M_w_analysis
    M_w_inv_yy_output = M_xx_output / det_M_w_analysis

    delta_k_perp_2 = 2 * np.sqrt(-1 / np.imag(M_w_inv_yy_output))
    delta_theta_m = np.sqrt(
        np.imag(M_w_inv_yy_output)
        / (
            (np.imag(M_w_inv_xy_output)) ** 2
            - np.imag(M_w_inv_xx_output) * np.imag(M_w_inv_yy_output)
        )
    ) / (K_magnitude_array)
    loc_m = np.exp(-2 * (theta_m_output / delta_theta_m) ** 2)

    print("polflux: ", poloidal_flux_output[cutoff_index])

    print("theta_m", theta_m_output[cutoff_index])
    print("delta_theta_m", delta_theta_m[cutoff_index])
    print(
        "mismatch attenuation",
        np.exp(-2 * (theta_m_output[cutoff_index] / delta_theta_m[cutoff_index]) ** 2),
    )
    # -----

    #    print(cutoff_index)

    # This part is used to make some nice plots when post-processing
    R_midplane_points = np.linspace(field.R_coord[0], field.R_coord[-1], 1000)
    # poloidal flux at R and z=0
    poloidal_flux_on_midplane = field.poloidal_flux(R_midplane_points, 0)

    # Calculates localisation (start)
    # Ray piece of localisation as a function of distance along ray
    K_magnitude_array_plus_KR = np.sqrt(
        (K_R_array + delta_K_R) ** 2
        + K_Z_array**2
        + K_zeta_initial**2 / q_R_array**2
    )
    K_magnitude_array_minus_KR = np.sqrt(
        (K_R_array - delta_K_R) ** 2
        + K_Z_array**2
        + K_zeta_initial**2 / q_R_array**2
    )
    K_magnitude_array_plus_Kzeta = np.sqrt(
        K_R_array**2
        + K_Z_array**2
        + (K_zeta_initial + delta_K_zeta) ** 2 / q_R_array**2
    )
    K_magnitude_array_minus_Kzeta = np.sqrt(
        K_R_array**2
        + K_Z_array**2
        + (K_zeta_initial - delta_K_zeta) ** 2 / q_R_array**2
    )
    K_magnitude_array_plus_KZ = np.sqrt(
        K_R_array**2
        + (K_Z_array + delta_K_Z) ** 2
        + K_zeta_initial**2 / q_R_array**2
    )
    K_magnitude_array_minus_KZ = np.sqrt(
        K_R_array**2
        + (K_Z_array - delta_K_Z) ** 2
        + K_zeta_initial**2 / q_R_array**2
    )

    H_1_Cardano_array, H_2_Cardano_array, H_3_Cardano_array = find_H_Cardano(
        K_magnitude_array,
        launch_angular_frequency,
        epsilon_para_output,
        epsilon_perp_output,
        epsilon_g_output,
        theta_m_output,
    )

    # In my experience, the H_3_Cardano expression corresponds to the O mode, and the H_2_Cardano expression corresponds to the X-mode
    # ALERT: This may not always be the case! Check the output figure to make sure that the appropriate solution is indeed 0 along the ray
    if mode_flag == 1:
        _, _, H_Cardano_plus_KR_array = find_H_Cardano(
            K_magnitude_array_plus_KR,
            launch_angular_frequency,
            epsilon_para_output,
            epsilon_perp_output,
            epsilon_g_output,
            theta_m_output,
        )
        _, _, H_Cardano_minus_KR_array = find_H_Cardano(
            K_magnitude_array_minus_KR,
            launch_angular_frequency,
            epsilon_para_output,
            epsilon_perp_output,
            epsilon_g_output,
            theta_m_output,
        )
        _, _, H_Cardano_plus_Kzeta_array = find_H_Cardano(
            K_magnitude_array_plus_Kzeta,
            launch_angular_frequency,
            epsilon_para_output,
            epsilon_perp_output,
            epsilon_g_output,
            theta_m_output,
        )
        _, _, H_Cardano_minus_Kzeta_array = find_H_Cardano(
            K_magnitude_array_minus_Kzeta,
            launch_angular_frequency,
            epsilon_para_output,
            epsilon_perp_output,
            epsilon_g_output,
            theta_m_output,
        )
        _, _, H_Cardano_plus_KZ_array = find_H_Cardano(
            K_magnitude_array_plus_KZ,
            launch_angular_frequency,
            epsilon_para_output,
            epsilon_perp_output,
            epsilon_g_output,
            theta_m_output,
        )
        _, _, H_Cardano_minus_KZ_array = find_H_Cardano(
            K_magnitude_array_minus_KZ,
            launch_angular_frequency,
            epsilon_para_output,
            epsilon_perp_output,
            epsilon_g_output,
            theta_m_output,
        )
    elif mode_flag == -1:
        _, H_Cardano_plus_KR_array, _ = find_H_Cardano(
            K_magnitude_array_plus_KR,
            launch_angular_frequency,
            epsilon_para_output,
            epsilon_perp_output,
            epsilon_g_output,
            theta_m_output,
        )
        _, H_Cardano_minus_KR_array, _ = find_H_Cardano(
            K_magnitude_array_minus_KR,
            launch_angular_frequency,
            epsilon_para_output,
            epsilon_perp_output,
            epsilon_g_output,
            theta_m_output,
        )
        _, H_Cardano_plus_Kzeta_array, _ = find_H_Cardano(
            K_magnitude_array_plus_Kzeta,
            launch_angular_frequency,
            epsilon_para_output,
            epsilon_perp_output,
            epsilon_g_output,
            theta_m_output,
        )
        _, H_Cardano_minus_Kzeta_array, _ = find_H_Cardano(
            K_magnitude_array_minus_Kzeta,
            launch_angular_frequency,
            epsilon_para_output,
            epsilon_perp_output,
            epsilon_g_output,
            theta_m_output,
        )
        _, H_Cardano_plus_KZ_array, _ = find_H_Cardano(
            K_magnitude_array_plus_KZ,
            launch_angular_frequency,
            epsilon_para_output,
            epsilon_perp_output,
            epsilon_g_output,
            theta_m_output,
        )
        _, H_Cardano_minus_KZ_array, _ = find_H_Cardano(
            K_magnitude_array_minus_KZ,
            launch_angular_frequency,
            epsilon_para_output,
            epsilon_perp_output,
            epsilon_g_output,
            theta_m_output,
        )

    g_R_Cardano = np.real(H_Cardano_plus_KR_array - H_Cardano_minus_KR_array) / (
        2 * delta_K_R
    )
    g_zeta_Cardano = np.real(
        H_Cardano_plus_Kzeta_array - H_Cardano_minus_Kzeta_array
    ) / (2 * delta_K_zeta)
    g_Z_Cardano = np.real(H_Cardano_plus_KZ_array - H_Cardano_minus_KZ_array) / (
        2 * delta_K_Z
    )

    g_magnitude_Cardano = np.sqrt(
        g_R_Cardano**2 + g_zeta_Cardano**2 + g_Z_Cardano**2
    )

    ##
    # From here on, we use the shorthand
    # loc: localisation
    # l_lc: distance from cutoff (l - l_c). Distance along the ray
    # cum: cumulative. As such, cum_loc is the cumulative integral of the localisation
    # p: polarisation
    # r: ray
    # b: beam
    # s: spectrum
    # Otherwise, variable names get really unwieldly
    ##

    # localisation_ray = g_magnitude_Cardano[0]**2/g_magnitude_Cardano**2
    # The first point of the beam may be very slightly in the plasma, so I have used the vacuum expression for the group velocity instead
    loc_r = (2 * constants.c / launch_angular_frequency) ** 2 / g_magnitude_Cardano**2

    # Spectrum piece of localisation as a function of distance along ray
    spectrum_power_law_coefficient = 13 / 3  # Turbulence cascade
    loc_s = (k_perp_1_bs / (-2 * wavenumber_K0)) ** (-spectrum_power_law_coefficient)

    # Beam piece of localisation as a function of distance along ray
    det_imag_Psi_w_analysis = (
        np.imag(Psi_xx_output) * np.imag(Psi_yy_output) - np.imag(Psi_xy_output) ** 2
    )  # Determinant of the imaginary part of Psi_w
    # Determinant of the real part of Psi_w. Not needed for the calculation, but gives useful insight
    det_real_Psi_w_analysis = (
        np.real(Psi_xx_output) * np.real(Psi_yy_output) - np.real(Psi_xy_output) ** 2
    )

    # Assumes circular beam at launch
    beam_waist_y = find_waist(launch_beam_width, wavenumber_K0, launch_beam_curvature)

    loc_b = (
        (beam_waist_y / np.sqrt(2))
        * det_imag_Psi_w_analysis
        / (abs(det_M_w_analysis) * np.sqrt(-np.imag(M_w_inv_yy_output)))
    )
    # --

    # Polarisation piece of localisation as a function of distance along ray
    # Polarisation e
    # eigenvector corresponding to eigenvalue = 0 (H=0)
    # First, find the components of the tensor D
    # Refer to 21st Dec 2020 notes for more
    # Note that e \cdot e* = 1
    [
        D_11_component,
        D_22_component,
        D_bb_component,
        D_12_component,
        D_1b_component,
    ] = find_D(
        K_magnitude_array,
        launch_angular_frequency,
        epsilon_para_output,
        epsilon_perp_output,
        epsilon_g_output,
        theta_m_output,
    )

    # Dispersion tensor
    D_tensor = np.zeros([numberOfDataPoints, 3, 3], dtype="complex128")
    D_tensor[:, 0, 0] = D_11_component
    D_tensor[:, 1, 1] = D_22_component
    D_tensor[:, 2, 2] = D_bb_component
    D_tensor[:, 0, 1] = -1j * D_12_component
    D_tensor[:, 1, 0] = 1j * D_12_component
    D_tensor[:, 0, 2] = D_1b_component
    D_tensor[:, 2, 0] = D_1b_component

    H_eigvals, e_eigvecs = np.linalg.eigh(D_tensor)

    # In my experience, H_eigvals[:,1] corresponds to the O mode, and H_eigvals[:,1] corresponds to the X-mode
    # ALERT: This may not always be the case! Check the output figure to make sure that the appropriate solution is indeed 0 along the ray
    # e_hat has components e_1,e_2,e_b
    if mode_flag == 1:
        H_solver = H_eigvals[:, 1]
        e_hat_output = e_eigvecs[:, :, 1]
    elif mode_flag == -1:
        H_solver = H_eigvals[:, 0]
        e_hat_output = e_eigvecs[:, :, 0]

    # equilibrium dielectric tensor - identity matrix. \bm{\epsilon}_{eq} - \bm{1}
    epsilon_minus_identity = np.zeros([numberOfDataPoints, 3, 3], dtype="complex128")
    epsilon_minus_identity[:, 0, 0] = epsilon_perp_output - np.ones(numberOfDataPoints)
    epsilon_minus_identity[:, 1, 1] = epsilon_perp_output - np.ones(numberOfDataPoints)
    epsilon_minus_identity[:, 2, 2] = epsilon_para_output - np.ones(numberOfDataPoints)
    epsilon_minus_identity[:, 0, 1] = -1j * epsilon_g_output
    epsilon_minus_identity[:, 1, 0] = 1j * epsilon_g_output

    # loc_p_unnormalised = abs(contract_special(np.conjugate(e_hat_output), contract_special(epsilon_minus_identity,e_hat_output)))**2 / (electron_density_output*10**19)**2

    # Avoids dividing a small number by another small number, leading to a big number because of numerical errors or something
    loc_p_unnormalised = np.divide(
        abs(
            contract_special(
                np.conjugate(e_hat_output),
                contract_special(epsilon_minus_identity, e_hat_output),
            )
        )
        ** 2,
        (electron_density_output * 10**19) ** 2,
        out=np.zeros_like(electron_density_output),
        where=electron_density_output > 1e-6,
    )
    loc_p = (
        launch_angular_frequency**2
        * constants.epsilon_0
        * find_electron_mass(temperature_output)
        / constants.e**2
    ) ** 2 * loc_p_unnormalised
    # Note that loc_p is called varepsilon in my paper

    # Note that K_1 = K cos theta_m, K_2 = 0, K_b = K sin theta_m, as a result of cold plasma dispersion
    K_hat_dot_e_hat = e_hat_output[:, 0] * np.cos(theta_m_output) + e_hat_output[
        :, 2
    ] * np.sin(theta_m_output)

    K_hat_dot_e_hat_sq = np.conjugate(K_hat_dot_e_hat) * K_hat_dot_e_hat
    # --

    # TODO: Come back and see if the naming of variables makes sense and is consistent

    l_lc = (
        distance_along_line - distance_along_line[cutoff_index]
    )  # Distance from cutoff

    # plt.figure()
    # plt.plot(l_lc,theta_m_output)
    # plt.axhline(constants.e**4 / (launch_angular_frequency**2 *constants.epsilon_0*constants.m_e)**2,c='k')

    # Combining the various localisation pieces to get some overall localisation
    # loc_p_r_s   =                  loc_p * loc_r * loc_s
    loc_b_r_s = loc_b * loc_r * loc_s
    loc_b_r = loc_b * loc_r

    if detailed_analysis_flag and (cutoff_index + 1 != len(tau_array)):
        """
        Now to do some more-complex analysis of the localisation.
        This part of the code fails in some situations, hence I'm making
        it possible to skip this section
        """
        # Finds the 1/e2 values (localisation)
        loc_b_r_s_max_over_e2 = (
            loc_b_r_s.max() / (np.e) ** 2
        )  # loc_b_r_s.max() / 2.71**2
        loc_b_r_max_over_e2 = loc_b_r.max() / (np.e) ** 2  # loc_b_r.max() / 2.71**2

        # Gives the inter-e2 range (analogous to interquartile range) in l-lc
        loc_b_r_s_delta_l_1 = find_x0(
            l_lc[0:cutoff_index], loc_b_r_s[0:cutoff_index], loc_b_r_s_max_over_e2
        )
        loc_b_r_s_delta_l_2 = find_x0(
            l_lc[cutoff_index::], loc_b_r_s[cutoff_index::], loc_b_r_s_max_over_e2
        )
        # The 1/e2 distances,  (l - l_c)
        loc_b_r_s_delta_l = np.array([loc_b_r_s_delta_l_1, loc_b_r_s_delta_l_2])
        loc_b_r_s_half_width_l = (loc_b_r_s_delta_l_2 - loc_b_r_s_delta_l_1) / 2
        loc_b_r_delta_l_1 = find_x0(
            l_lc[0:cutoff_index], loc_b_r[0:cutoff_index], loc_b_r_max_over_e2
        )
        loc_b_r_delta_l_2 = find_x0(
            l_lc[cutoff_index::], loc_b_r[cutoff_index::], loc_b_r_max_over_e2
        )
        # The 1/e2 distances,  (l - l_c)
        loc_b_r_delta_l = np.array([loc_b_r_delta_l_1, loc_b_r_delta_l_2])
        loc_b_r_half_width_l = (loc_b_r_delta_l_1 - loc_b_r_delta_l_2) / 2

        # Estimates the inter-e2 range (analogous to interquartile range) in kperp1, from l-lc
        # Bear in mind that since abs(kperp1) is minimised at cutoff, one really has to use that in addition to these.
        loc_b_r_s_delta_kperp1_1 = find_x0(
            k_perp_1_bs[0:cutoff_index], l_lc[0:cutoff_index], loc_b_r_s_delta_l_1
        )
        loc_b_r_s_delta_kperp1_2 = find_x0(
            k_perp_1_bs[cutoff_index::], l_lc[cutoff_index::], loc_b_r_s_delta_l_2
        )
        loc_b_r_s_delta_kperp1 = np.array(
            [loc_b_r_s_delta_kperp1_1, loc_b_r_s_delta_kperp1_2]
        )
        loc_b_r_delta_kperp1_1 = find_x0(
            k_perp_1_bs[0:cutoff_index], l_lc[0:cutoff_index], loc_b_r_delta_l_1
        )
        loc_b_r_delta_kperp1_2 = find_x0(
            k_perp_1_bs[cutoff_index::], l_lc[cutoff_index::], loc_b_r_delta_l_2
        )
        loc_b_r_delta_kperp1 = np.array(
            [loc_b_r_delta_kperp1_1, loc_b_r_delta_kperp1_2]
        )

        # Calculate the cumulative integral of the localisation pieces
        cum_loc_b_r_s = integrate.cumtrapz(loc_b_r_s, distance_along_line, initial=0)
        cum_loc_b_r_s = cum_loc_b_r_s - max(cum_loc_b_r_s) / 2
        cum_loc_b_r = integrate.cumtrapz(loc_b_r, distance_along_line, initial=0)
        cum_loc_b_r = cum_loc_b_r - max(cum_loc_b_r) / 2

        # Finds the 1/e2 values (cumulative integral of localisation)
        # cum_loc_b_r_s_max_over_e2_1 = cum_loc_b_r_s.min() * (1 - 1 / (np.e)**2)
        # cum_loc_b_r_s_max_over_e2_2 = cum_loc_b_r_s.max() * (1 - 1 / (np.e)**2)
        # cum_loc_b_r_max_over_e2_1 = cum_loc_b_r.min() * (1 - 1 / (np.e)**2)
        # cum_loc_b_r_max_over_e2_2 = cum_loc_b_r.max() * (1 - 1 / (np.e)**2)
        cum_loc_b_r_s_max_over_e2 = cum_loc_b_r_s.max() * (1 - 1 / (np.e) ** 2)
        cum_loc_b_r_max_over_e2 = cum_loc_b_r.max() * (1 - 1 / (np.e) ** 2)

        # Gives the inter-e range (analogous to interquartile range) in l-lc
        cum_loc_b_r_s_delta_l_1 = find_x0(
            l_lc, cum_loc_b_r_s, -cum_loc_b_r_s_max_over_e2
        )
        cum_loc_b_r_s_delta_l_2 = find_x0(
            l_lc, cum_loc_b_r_s, cum_loc_b_r_s_max_over_e2
        )
        cum_loc_b_r_s_delta_l = np.array(
            [cum_loc_b_r_s_delta_l_1, cum_loc_b_r_s_delta_l_2]
        )
        cum_loc_b_r_s_half_width = (
            cum_loc_b_r_s_delta_l_2 - cum_loc_b_r_s_delta_l_1
        ) / 2
        cum_loc_b_r_delta_l_1 = find_x0(l_lc, cum_loc_b_r, -cum_loc_b_r_max_over_e2)
        cum_loc_b_r_delta_l_2 = find_x0(l_lc, cum_loc_b_r, cum_loc_b_r_max_over_e2)
        cum_loc_b_r_delta_l = np.array([cum_loc_b_r_delta_l_1, cum_loc_b_r_delta_l_2])
        cum_loc_b_r_half_width = (cum_loc_b_r_delta_l_2 - cum_loc_b_r_delta_l_1) / 2

        # Gives the inter-e2 range (analogous to interquartile range) in kperp1.
        # Bear in mind that since abs(kperp1) is minimised at cutoff, one really has to use that in addition to these.
        cum_loc_b_r_s_delta_kperp1_1 = find_x0(
            k_perp_1_bs[0:cutoff_index],
            cum_loc_b_r_s[0:cutoff_index],
            -cum_loc_b_r_s_max_over_e2,
        )
        cum_loc_b_r_s_delta_kperp1_2 = find_x0(
            k_perp_1_bs[cutoff_index::],
            cum_loc_b_r_s[cutoff_index::],
            cum_loc_b_r_s_max_over_e2,
        )
        cum_loc_b_r_s_delta_kperp1 = np.array(
            [cum_loc_b_r_s_delta_kperp1_1, cum_loc_b_r_s_delta_kperp1_2]
        )
        cum_loc_b_r_delta_kperp1_1 = find_x0(
            k_perp_1_bs[0:cutoff_index],
            cum_loc_b_r[0:cutoff_index],
            -cum_loc_b_r_max_over_e2,
        )
        cum_loc_b_r_delta_kperp1_2 = find_x0(
            k_perp_1_bs[cutoff_index::],
            cum_loc_b_r[cutoff_index::],
            cum_loc_b_r_max_over_e2,
        )
        cum_loc_b_r_delta_kperp1 = np.array(
            [cum_loc_b_r_delta_kperp1_1, cum_loc_b_r_delta_kperp1_2]
        )

        # Gives the mode l-lc for backscattering
        loc_b_r_s_max_index = find_nearest(loc_b_r_s, loc_b_r_s.max())
        loc_b_r_s_max_l_lc = (
            distance_along_line[loc_b_r_s_max_index] - distance_along_line[cutoff_index]
        )
        loc_b_r_max_index = find_nearest(loc_b_r, loc_b_r.max())
        loc_b_r_max_l_lc = (
            distance_along_line[loc_b_r_max_index] - distance_along_line[cutoff_index]
        )

        # Gives the mean l-lc for backscattering
        cum_loc_b_r_s_mean_l_lc = (
            np.trapz(loc_b_r_s * distance_along_line, distance_along_line)
            / np.trapz(loc_b_r_s, distance_along_line)
            - distance_along_line[cutoff_index]
        )
        cum_loc_b_r_mean_l_lc = (
            np.trapz(loc_b_r * distance_along_line, distance_along_line)
            / np.trapz(loc_b_r, distance_along_line)
            - distance_along_line[cutoff_index]
        )

        # Gives the median l-lc for backscattering
        cum_loc_b_r_s_delta_l_0 = find_x0(l_lc, cum_loc_b_r_s, 0)
        cum_loc_b_r_delta_l_0 = find_x0(l_lc, cum_loc_b_r, 0)

        # Due to the divergency of the ray piece, the mode kperp1 for backscattering is exactly that at the cut-off

        # Gives the mean kperp1 for backscattering
        cum_loc_b_r_s_mean_kperp1 = np.trapz(
            loc_b_r_s * k_perp_1_bs, k_perp_1_bs
        ) / np.trapz(loc_b_r_s, k_perp_1_bs)
        cum_loc_b_r_mean_kperp1 = np.trapz(
            loc_b_r * k_perp_1_bs, k_perp_1_bs
        ) / np.trapz(loc_b_r, k_perp_1_bs)

        # Gives the median kperp1 for backscattering
        cum_loc_b_r_s_delta_kperp1_0 = find_x0(k_perp_1_bs, cum_loc_b_r_s, 0)
        # Only works if point is before cutoff. To fix.
        cum_loc_b_r_delta_kperp1_0 = find_x0(
            k_perp_1_bs[0:cutoff_index], cum_loc_b_r[0:cutoff_index], 0
        )

        # To make the plots look nice
        k_perp_1_bs_plot = np.append(-2 * wavenumber_K0, k_perp_1_bs)
        k_perp_1_bs_plot = np.append(k_perp_1_bs_plot, -2 * wavenumber_K0)
        cum_loc_b_r_s_plot = np.append(cum_loc_b_r_s[0], cum_loc_b_r_s)
        cum_loc_b_r_s_plot = np.append(cum_loc_b_r_s_plot, cum_loc_b_r_s[-1])
        cum_loc_b_r_plot = np.append(cum_loc_b_r[0], cum_loc_b_r)
        cum_loc_b_r_plot = np.append(cum_loc_b_r_plot, cum_loc_b_r[-1])
    else:
        loc_b_r_s_max_over_e2 = None
        loc_b_r_max_over_e2 = None
        loc_b_r_s_delta_l = None
        loc_b_r_delta_l = None
        loc_b_r_s_delta_kperp1 = None
        loc_b_r_delta_kperp1 = None
        cum_loc_b_r_s = None
        cum_loc_b_r = None
        k_perp_1_bs_plot = None
        cum_loc_b_r_s_plot = None
        cum_loc_b_r_plot = None
        cum_loc_b_r_s_max_over_e2 = None
        cum_loc_b_r_max_over_e2 = None
        cum_loc_b_r_s_delta_l = None
        cum_loc_b_r_delta_l = None
        cum_loc_b_r_s_delta_kperp1 = None
        cum_loc_b_r_delta_kperp1 = None
        loc_b_r_s_max_l_lc = None
        loc_b_r_max_l_lc = None
        cum_loc_b_r_s_mean_l_lc = None
        cum_loc_b_r_mean_l_lc = None
        cum_loc_b_r_s_delta_l_0 = None
        cum_loc_b_r_delta_l_0 = None
        cum_loc_b_r_s_mean_kperp1 = None
        cum_loc_b_r_mean_kperp1 = None
        cum_loc_b_r_s_delta_kperp1_0 = None
        cum_loc_b_r_delta_kperp1_0 = None

    # integrated_localisation_b_p_r_delta_kperp1_0 = find_x0(k_perp_1_bs[0:cutoff_index],integrated_localisation_b_p_r[0:cutoff_index],0)

    #     # -------------------

    # Calculates localisation (relevant pieces of the Spherical Tokamak case)
    d_theta_m_d_tau = np.gradient(theta_m_output, tau_array)
    d_K_d_tau = np.gradient(K_magnitude_array, tau_array)
    d_tau_B_d_tau_C = (
        g_magnitude_Cardano / g_magnitude_output
    )  # d tau_Booker / d tau_Cardano
    theta_m_min_idx = np.argmin(abs(theta_m_output))
    delta_kperp1_ST = k_perp_1_bs - k_perp_1_bs[theta_m_min_idx]
    G_full = (
        (
            d_K_d_tau * g_magnitude_output
            - K_magnitude_array**2 * d_theta_m_d_tau**2 * M_w_inv_xx_output
        )
        * d_tau_B_d_tau_C**2
    ) ** (-1)
    G_term1 = (d_K_d_tau * g_magnitude_output * d_tau_B_d_tau_C**2) ** (-1)
    G_term2 = (
        K_magnitude_array**2
        * d_theta_m_d_tau**2
        * M_w_inv_xx_output
        * G_term1**2
        * d_tau_B_d_tau_C**2
    ) ** (-1)
    # print('ST 1st term: ', G_term1[theta_m_min_idx])
    # print('ST 2nd term: ', G_term2[theta_m_min_idx])
    # print('ST full: ', G_full[theta_m_min_idx])
    # print('ST 2nd term / ST 1st term: ', abs(G_term2[theta_m_min_idx]/G_term1[theta_m_min_idx]))
    # print('ST first 2 terms / ST full: ', abs((G_term2[theta_m_min_idx]+G_term1[theta_m_min_idx])/G_full[theta_m_min_idx]) )

    # Calculates nabla nabla H, nabla_K nabla H, nabla_K nabla_K H
    grad_grad_H, gradK_grad_H, gradK_gradK_H = hessians(dH)

    # gradK_gradK_H[:,0,1] = - gradK_gradK_H[:,0,1]

    # ## Mode conversion stuff
    # def find_N(Booker_alpha,Booker_beta,Booker_gamma,mode_flag):
    #     N = np.sqrt( - (
    #             Booker_beta - mode_flag *
    #             np.sqrt(np.maximum(
    #                     np.zeros_like(Booker_beta),
    #                     (Booker_beta**2 - 4*Booker_alpha*Booker_gamma)
    #                 )
    #             )
    #             # np.sqrt(Booker_beta**2 - 4*Booker_alpha*Booker_gamma)
    #             ) / (2 * Booker_alpha) )

    #     return N

    # #
    # sin_theta_m_sq = (np.sin(theta_m_output))**2
    # Booker_alpha = epsilon_para_output*sin_theta_m_sq + epsilon_perp_output*(1-sin_theta_m_sq)
    # Booker_beta = (
    #         - epsilon_perp_output * epsilon_para_output * (1+sin_theta_m_sq)
    #         - (epsilon_perp_output**2 - epsilon_g_output**2) * (1-sin_theta_m_sq)
    #                 )
    # Booker_gamma = epsilon_para_output*(epsilon_perp_output**2 - epsilon_g_output**2)

    # N_X = find_N(Booker_alpha,Booker_beta,Booker_gamma,-1)
    # N_O = find_N(Booker_alpha,Booker_beta,Booker_gamma,1)

    # B_Z_plus = find_B_Z(q_R_array+delta_R,q_Z_array)
    # B_T_plus = find_B_T(q_R_array+delta_R,q_Z_array)
    # B_Z_minus = find_B_Z(q_R_array-delta_R,q_Z_array)
    # B_T_minus = find_B_T(q_R_array-delta_R,q_Z_array)

    # pitch_angle = np.arctan2(B_Z_output,B_T_output)
    # pitch_angle_plus = np.arctan2(B_Z_plus,B_T_plus)
    # pitch_angle_minus = np.arctan2(B_Z_minus,B_T_minus)

    # d_pitch_angle_d_R = (pitch_angle_plus - pitch_angle_minus) / 2 * delta_R

    # # ---
    # anisotropy_term = (1 - N_O**2 / N_X**2)
    # shear_term = (1/(4 * wavenumber_K0)) * d_pitch_angle_d_R
    # print((abs(anisotropy_term)-abs(shear_term)))
    # plt.figure()
    # plt.plot(l_lc, (abs(anisotropy_term)-abs(shear_term)))
    # ##

    # Running some tests

    # plt.figure()
    # plt.plot(l_lc, H_output)

    # g_hat_Cartesian = np.zeros([numberOfDataPoints,3])
    # g_hat_Cartesian[:,0] = g_hat_output[:,0]*np.cos(q_zeta_array ) - g_hat_output[:,1]*np.sin(q_zeta_array )
    # g_hat_Cartesian[:,1] = g_hat_output[:,0]*np.sin(q_zeta_array ) + g_hat_output[:,1]*np.cos(q_zeta_array )
    # g_hat_Cartesian[:,2] = g_hat_output[:,2]

    # Psi_xg_output = contract_special(x_hat_Cartesian,contract_special(Psi_3D_Cartesian,g_hat_Cartesian))
    # Psi_yg_output = contract_special(y_hat_Cartesian,contract_special(Psi_3D_Cartesian,g_hat_Cartesian))
    # Psi_gg_output = contract_special(g_hat_Cartesian,contract_special(Psi_3D_Cartesian,g_hat_Cartesian))

    # plt.figure()
    # plt.plot(l_lc, np.imag(Psi_xg_output)/det_imag_Psi_w_analysis)
    # plt.plot(l_lc, np.imag(Psi_yg_output)/det_imag_Psi_w_analysis)
    # plt.plot(l_lc, np.imag(Psi_gg_output)/det_imag_Psi_w_analysis)

    # Psi_3D_test = np.zeros_like(Psi_3D_output,dtype='complex128')
    # d_Psi_d_tau_all = np.zeros_like(Psi_3D_output,dtype='complex128')

    # Psi_3D_test[0,:,:] = Psi_3D_output[0,:,:]

    # for ii in range(1,len(q_R_array)):
    #     d_Psi_d_tau = ( - grad_grad_H[ii-1,:,:]
    #             - np.matmul(
    #                         Psi_3D_test[ii-1,:,:], gradK_grad_H[ii-1,:,:]
    #                         )
    #             - np.matmul(
    #                         grad_gradK_H[ii-1,:,:], Psi_3D_test[ii-1,:,:]
    #                         )
    #             - np.matmul(np.matmul(
    #                         Psi_3D_test[ii-1,:,:], gradK_gradK_H[ii-1,:,:]
    #                         ),
    #                         Psi_3D_test[ii-1,:,:]
    #                         )
    #         )
    #     d_Psi_d_tau_all[ii-1,:,:] = d_Psi_d_tau

    #     if  ii<3:
    #         Psi_3D_test[ii,:,:] =  Psi_3D_test[ii-1,:,:] + (tau_array[ii] - tau_array[ii-1]) * d_Psi_d_tau_all[ii-1,:,:]
    #     else:
    #         Psi_3D_test[ii,:,:] =  Psi_3D_test[ii-1,:,:] + (tau_array[ii] - tau_array[ii-1]) * (
    #                   (23/12) * d_Psi_d_tau_all[ii-1,:,:]
    #                 - (16/12) * d_Psi_d_tau_all[ii-2,:,:]
    #                 + (5/12)  * d_Psi_d_tau_all[ii-3,:,:]
    #             )

    # Psi_3D_Cartesian_test = find_Psi_3D_lab_Cartesian(Psi_3D_test, q_R_array, q_zeta_array, K_R_array, K_zeta_initial)
    # Psi_xx_test = contract_special(x_hat_Cartesian,contract_special(Psi_3D_Cartesian_test,x_hat_Cartesian))
    # Psi_xy_test = contract_special(x_hat_Cartesian,contract_special(Psi_3D_Cartesian_test,y_hat_Cartesian))
    # Psi_yy_test = contract_special(y_hat_Cartesian,contract_special(Psi_3D_Cartesian_test,y_hat_Cartesian))

    # plt.figure()
    # plt.plot(l_lc,np.imag(Psi_3D_test[:,0,0]),'r')
    # plt.plot(l_lc,np.imag(Psi_3D_output[:,0,0]),'k')

    # plt.figure()
    # plt.subplot(2,3,1)
    # plt.plot(l_lc,np.imag(Psi_xx_test),'r')
    # plt.plot(l_lc,np.imag(Psi_xx_output),'k')
    # plt.subplot(2,3,2)
    # plt.plot(l_lc,np.imag(Psi_xy_test),'r')
    # plt.plot(l_lc,np.imag(Psi_xy_output),'k')
    # plt.subplot(2,3,3)
    # plt.plot(l_lc,np.imag(Psi_yy_test),'r')
    # plt.plot(l_lc,np.imag(Psi_yy_output),'k')
    # plt.subplot(2,3,4)
    # plt.plot(l_lc,np.real(Psi_xx_test),'r')
    # plt.plot(l_lc,np.real(Psi_xx_output),'k')
    # plt.subplot(2,3,5)
    # plt.plot(l_lc,np.real(Psi_xy_test),'r')
    # plt.plot(l_lc,np.real(Psi_xy_output),'k')
    # plt.subplot(2,3,6)
    # plt.plot(l_lc,np.real(Psi_yy_test),'r')
    # plt.plot(l_lc,np.real(Psi_yy_output),'k')

    # plt.figure()
    # plt.subplot(3,3,1)
    # plt.plot(l_lc,gradK_gradK_H[:,0,0],'r')
    # plt.subplot(3,3,2)
    # plt.plot(l_lc,gradK_gradK_H[:,1,0],'r')
    # plt.subplot(3,3,3)
    # plt.plot(l_lc,gradK_gradK_H[:,2,0],'r')
    # plt.subplot(3,3,5)
    # plt.plot(l_lc,gradK_gradK_H[:,1,1],'r')
    # plt.subplot(3,3,6)
    # plt.plot(l_lc,gradK_gradK_H[:,1,2],'r')
    # plt.subplot(3,3,9)
    # plt.plot(l_lc,gradK_gradK_H[:,2,2],'r')
    ##

    # plt.figure()
    # plt.subplot(2,2,1)
    # plt.plot(l_lc,loc_ST_decay,label=r'$\exp [ \frac{1}{2} Im [ (\Delta k_{\perp,1} K_0 \frac{d \theta_m}{d \tau})^2 M_{xx,0}^{-1} (\frac{d K}{d \tau})^{-2} ] ]$')
    # plt.legend(fontsize=6)
    # plt.subplot(2,2,2)
    # plt.plot(l_lc,theta_m_output,label=r'$\theta_{m}$')
    # plt.legend()
    # plt.subplot(2,2,3)
    # plt.plot(l_lc,delta_kperp1_ST,label=r'$\Delta k_{\perp,1}$')
    # plt.legend()
    # plt.subplot(2,2,4)
    # plt.plot(l_lc,loc_m,label=r'$loc_m$')
    # plt.legend()
    # plt.gcf().set_dpi(300)

    # -------------------
    # This saves the data generated by the analysis after the main loop
    # -------------------
    print("Saving analysis data")
    np.savez(
        output_path / f"analysis_output{output_filename_suffix}",
        Psi_xx_output=Psi_xx_output,
        Psi_xy_output=Psi_xy_output,
        Psi_yy_output=Psi_yy_output,
        Psi_xg_output=Psi_xg_output,
        Psi_yg_output=Psi_yg_output,
        Psi_gg_output=Psi_gg_output,
        Psi_xx_entry=Psi_xx_entry,
        Psi_xy_entry=Psi_xy_entry,
        Psi_yy_entry=Psi_yy_entry,
        Psi_3D_Cartesian=Psi_3D_Cartesian,
        x_hat_Cartesian=x_hat_Cartesian,
        y_hat_Cartesian=y_hat_Cartesian,
        g_hat_Cartesian=g_hat_Cartesian,
        M_xx_output=M_xx_output,
        M_xy_output=M_xy_output,
        M_yy_output=M_yy_output,
        M_w_inv_xx_output=M_w_inv_xx_output,
        M_w_inv_xy_output=M_w_inv_xy_output,
        M_w_inv_yy_output=M_w_inv_yy_output,
        xhat_dot_grad_bhat_dot_xhat_output=xhat_dot_grad_bhat_dot_xhat_output,
        xhat_dot_grad_bhat_dot_yhat_output=xhat_dot_grad_bhat_dot_yhat_output,
        xhat_dot_grad_bhat_dot_ghat_output=xhat_dot_grad_bhat_dot_ghat_output,
        yhat_dot_grad_bhat_dot_xhat_output=yhat_dot_grad_bhat_dot_xhat_output,
        yhat_dot_grad_bhat_dot_yhat_output=yhat_dot_grad_bhat_dot_yhat_output,
        yhat_dot_grad_bhat_dot_ghat_output=yhat_dot_grad_bhat_dot_ghat_output,
        grad_grad_H=grad_grad_H,
        gradK_grad_H=gradK_grad_H,
        gradK_gradK_H=gradK_gradK_H,
        d_theta_d_tau=d_theta_d_tau,
        d_xhat_d_tau_dot_yhat_output=d_xhat_d_tau_dot_yhat_output,
        kappa_dot_xhat_output=kappa_dot_xhat_output,
        kappa_dot_yhat_output=kappa_dot_yhat_output,
        kappa_dot_ghat_output=kappa_dot_ghat_output,
        kappa_magnitude=kappa_magnitude,
        delta_k_perp_2=delta_k_perp_2,
        delta_theta_m=delta_theta_m,
        theta_m_output=theta_m_output,
        RZ_distance_along_line=RZ_distance_along_line,
        distance_along_line=distance_along_line,
        k_perp_1_bs=k_perp_1_bs,
        K_magnitude_array=K_magnitude_array,
        cutoff_index=cutoff_index,
        x_hat_output=x_hat_output,
        y_hat_output=y_hat_output,
        b_hat_output=b_hat_output,
        g_hat_output=g_hat_output,
        e_hat_output=e_hat_output,
        H_eigvals=H_eigvals,
        e_eigvecs=e_eigvecs,
        H_1_Cardano_array=H_1_Cardano_array,
        H_2_Cardano_array=H_2_Cardano_array,
        H_3_Cardano_array=H_3_Cardano_array,
        kperp1_hat_output=kperp1_hat_output,
        theta_output=theta_output,
        g_magnitude_Cardano=g_magnitude_Cardano,
        # in_index=in_index,out_index=out_index,
        poloidal_flux_on_midplane=poloidal_flux_on_midplane,
        R_midplane_points=R_midplane_points,
        loc_b=loc_b,
        loc_p=loc_p,
        loc_r=loc_r,
        loc_s=loc_s,
        loc_m=loc_m,
        loc_b_r_s=loc_b_r_s,
        loc_b_r=loc_b_r,
        # Detailed analysis
        loc_b_r_s_max_over_e2=loc_b_r_s_max_over_e2,
        loc_b_r_max_over_e2=loc_b_r_max_over_e2,
        # The 1/e2 distances,  (l - l_c)
        loc_b_r_s_delta_l=loc_b_r_s_delta_l,
        loc_b_r_delta_l=loc_b_r_delta_l,
        # The 1/e2 distances, kperp1, estimated from (l - l_c)
        loc_b_r_s_delta_kperp1=loc_b_r_s_delta_kperp1,
        loc_b_r_delta_kperp1=loc_b_r_delta_kperp1,
        cum_loc_b_r_s=cum_loc_b_r_s,
        cum_loc_b_r=cum_loc_b_r,
        k_perp_1_bs_plot=k_perp_1_bs_plot,
        cum_loc_b_r_s_plot=cum_loc_b_r_s_plot,
        cum_loc_b_r_plot=cum_loc_b_r_plot,
        cum_loc_b_r_s_max_over_e2=cum_loc_b_r_s_max_over_e2,
        cum_loc_b_r_max_over_e2=cum_loc_b_r_max_over_e2,
        # The cumloc 1/e2 distances, (l - l_c)
        cum_loc_b_r_s_delta_l=cum_loc_b_r_s_delta_l,
        cum_loc_b_r_delta_l=cum_loc_b_r_delta_l,
        # The cumloc 1/e2 distances, kperp1
        cum_loc_b_r_s_delta_kperp1=cum_loc_b_r_s_delta_kperp1,
        cum_loc_b_r_delta_kperp1=cum_loc_b_r_delta_kperp1,
        loc_b_r_s_max_l_lc=loc_b_r_s_max_l_lc,
        loc_b_r_max_l_lc=loc_b_r_max_l_lc,  # mode l-lc
        cum_loc_b_r_s_mean_l_lc=cum_loc_b_r_s_mean_l_lc,
        cum_loc_b_r_mean_l_lc=cum_loc_b_r_mean_l_lc,  # mean l-lc
        cum_loc_b_r_s_delta_l_0=cum_loc_b_r_s_delta_l_0,
        cum_loc_b_r_delta_l_0=cum_loc_b_r_delta_l_0,  # median l-lc
        cum_loc_b_r_s_mean_kperp1=cum_loc_b_r_s_mean_kperp1,
        cum_loc_b_r_mean_kperp1=cum_loc_b_r_mean_kperp1,  # mean kperp1
        cum_loc_b_r_s_delta_kperp1_0=cum_loc_b_r_s_delta_kperp1_0,
        cum_loc_b_r_delta_kperp1_0=cum_loc_b_r_delta_kperp1_0,  # median kperp1
        # det_imag_Psi_w_analysis=det_imag_Psi_w_analysis,det_real_Psi_w_analysis=det_real_Psi_w_analysis,det_M_w_analysis=det_M_w_analysis
    )
    print("Analysis data saved")
    # -------------------

    # -------------------
    # Sanity check. Makes sure that calculated quantities are reasonable
    # -------------------
    check_output(H_output)
    # -------------------

    # -------------------
    # This saves some simple figures
    # Allows one to quickly gain an insight into what transpired in the simulation
    # -------------------
    if figure_flag:
        print("Making figures")
        output_figurename_suffix = output_filename_suffix + ".png"

        """
        Plots the beam path on the R Z plane
        """
        plt.figure()
        plt.title("Rz")
        plt.xlabel("R / m")  # x-direction
        plt.ylabel("z / m")

        contour_levels = np.linspace(0, 1.0, 11)
        CS = plt.contour(
            field.R_coord,
            field.Z_coord,
            np.transpose(field.poloidalFlux_grid),
            contour_levels,
            vmin=0,
            vmax=1,
            cmap="plasma_r",
        )
        plt.clabel(CS, inline=1, fontsize=10)  # Labels the flux surfaces
        plt.plot(
            np.concatenate([[launch_position[0], initial_position[0]], q_R_array]),
            np.concatenate([[launch_position[2], initial_position[2]], q_Z_array]),
            "--.k",
        )  # Central (reference) ray
        # cutoff_contour = plt.contour(x_grid, z_grid, normalised_plasma_freq_grid,
        #                             levels=1,vmin=1,vmax=1,linewidths=5,colors='grey')
        plt.xlim(field.R_coord[0], field.R_coord[-1])
        plt.ylim(field.Z_coord[0], field.Z_coord[-1])

        plt.savefig(output_path / f"Ray1_{output_figurename_suffix}")
        plt.close()

        """
        Plots Cardano's and np.linalg's solutions to the actual dispersion relation
        Useful to check whether the solution which = 0 along the path changes
        """
        plt.figure()
        plt.plot(l_lc, abs(H_eigvals[:, 0]), "ro")
        plt.plot(l_lc, abs(H_eigvals[:, 1]), "go")
        plt.plot(l_lc, abs(H_eigvals[:, 2]), "bo")
        plt.plot(l_lc, abs(H_1_Cardano_array), "r")
        plt.plot(l_lc, abs(H_2_Cardano_array), "g")
        plt.plot(l_lc, abs(H_3_Cardano_array), "b")
        plt.savefig(output_path / f"H_{output_figurename_suffix}")
        plt.close()

        # Commented out because this does not work properly
        # """
        # Plots Psi before and after the BCs are applied
        # """
        # K_magnitude_entry = np.sqrt(K_R_entry**2 + K_zeta_entry**2 * entry_position[0]**2 + K_Z_entry**2)

        # Psi_w_entry = np.array([
        # [Psi_xx_entry,Psi_xy_entry],
        # [Psi_xy_entry,Psi_yy_entry]
        # ])

        # Psi_w_initial = np.array([
        #         [Psi_xx_output[0],Psi_xy_output[0]],
        #         [Psi_xy_output[0],Psi_yy_output[0]]
        #         ])

        # [Psi_w_entry_real_eigval_a, Psi_w_entry_real_eigval_b], Psi_w_entry_real_eigvec = np.linalg.eig(np.real(Psi_w_entry))
        # [Psi_w_entry_imag_eigval_a, Psi_w_entry_imag_eigval_b], Psi_w_entry_imag_eigvec = np.linalg.eig(np.imag(Psi_w_entry))
        # Psi_w_entry_real_eigvec_a = Psi_w_entry_real_eigvec[:,0]
        # Psi_w_entry_real_eigvec_b = Psi_w_entry_real_eigvec[:,1]
        # Psi_w_entry_imag_eigvec_a = Psi_w_entry_imag_eigvec[:,0]
        # Psi_w_entry_imag_eigvec_b = Psi_w_entry_imag_eigvec[:,1]

        # [Psi_w_initial_real_eigval_a, Psi_w_initial_real_eigval_b], Psi_w_initial_real_eigvec = np.linalg.eig(np.real(Psi_w_initial))
        # [Psi_w_initial_imag_eigval_a, Psi_w_initial_imag_eigval_b], Psi_w_initial_imag_eigvec = np.linalg.eig(np.imag(Psi_w_initial))
        # Psi_w_initial_real_eigvec_a = Psi_w_initial_real_eigvec[:,0]
        # Psi_w_initial_real_eigvec_b = Psi_w_initial_real_eigvec[:,1]
        # Psi_w_initial_imag_eigvec_a = Psi_w_initial_imag_eigvec[:,0]
        # Psi_w_initial_imag_eigvec_b = Psi_w_initial_imag_eigvec[:,1]

        # numberOfPlotPoints = 50
        # sin_array = np.sin(np.linspace(0,2*np.pi,numberOfPlotPoints))
        # cos_array = np.cos(np.linspace(0,2*np.pi,numberOfPlotPoints))

        # width_ellipse_entry = np.zeros([numberOfPlotPoints,2])
        # width_ellipse_initial = np.zeros([numberOfPlotPoints,2])
        # rad_curv_ellipse_entry = np.zeros([numberOfPlotPoints,2])
        # rad_curv_ellipse_initial = np.zeros([numberOfPlotPoints,2])
        # for ii in range(0,numberOfPlotPoints):
        #     width_ellipse_entry[ii,:] = np.sqrt(2/Psi_w_entry_imag_eigval_a)*Psi_w_entry_imag_eigvec_a*sin_array[ii] + np.sqrt(2/Psi_w_entry_imag_eigval_b)*Psi_w_entry_imag_eigvec_b*cos_array[ii]
        #     width_ellipse_initial[ii,:] = np.sqrt(2/Psi_w_initial_imag_eigval_a)*Psi_w_initial_imag_eigvec_a*sin_array[ii] + np.sqrt(2/Psi_w_initial_imag_eigval_b)*Psi_w_initial_imag_eigvec_b*cos_array[ii]

        #     rad_curv_ellipse_entry[ii,:] = (K_magnitude_entry/Psi_w_entry_real_eigval_a)*Psi_w_entry_real_eigvec_a*sin_array[ii] + (K_magnitude_entry/Psi_w_entry_real_eigval_b)*Psi_w_entry_real_eigvec_b*cos_array[ii]
        #     rad_curv_ellipse_initial[ii,:] = (K_magnitude_array[0]/Psi_w_initial_real_eigval_a)*Psi_w_initial_real_eigvec_a*sin_array[ii] + (K_magnitude_array[0]/Psi_w_initial_real_eigval_b)*Psi_w_initial_real_eigvec_b*cos_array[ii]

        # plt.figure()
        # plt.subplot(1,2,1)
        # plt.plot(width_ellipse_entry[:,0],width_ellipse_entry[:,1])
        # plt.plot(width_ellipse_initial[:,0],width_ellipse_initial[:,1])
        # plt.gca().set_aspect('equal', adjustable='box')

        # plt.subplot(1,2,2)
        # plt.plot(rad_curv_ellipse_entry[:,0],rad_curv_ellipse_entry[:,1])
        # plt.plot(rad_curv_ellipse_initial[:,0],rad_curv_ellipse_initial[:,1])
        # plt.gca().set_aspect('equal', adjustable='box')
        # plt.savefig('BC_' + output_filename_suffix)
        # plt.close()

        print("Figures have been saved")
    # -------------------

    return None


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
