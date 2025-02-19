import datetime
import datatree
import json
import numpy as np
import pathlib
from scipy.integrate import solve_ivp
from scotty.fun_evolution_3D import pack_beam_parameters_3D, unpack_beam_parameters_3D, beam_evolution_fun_3D
from scotty.fun_general import freq_GHz_to_angular_frequency
from scotty.geometry_3D import MagneticField_3D_Cartesian, InterpolatedField_3D_Cartesian
from scotty.hamiltonian_3D import Hamiltonian_3D
from scotty.profile_fit import ProfileFitLike, profile_fit
from scotty.ray_solver_3D import propagate_ray
from scotty.typing import ArrayLike, FloatArray, PathLike
from scotty._version import __version__
import time
from typing import cast, Optional, Sequence, Union
import uuid
import xarray as xr

def beam_me_up_3D(
    poloidal_launch_angle_Torbeam: float,
    toroidal_launch_angle_Torbeam: float,
    launch_freq_GHz: float,
    mode_flag: int,
    launch_beam_width: float,
    launch_beam_curvature: float,
    launch_position_cartesian: FloatArray,

    # keyword arguments begin
    vacuumLaunch_flag: bool = True,
    relativistic_flag: bool = False,  # includes relativistic corrections to electron mass when set to True
    find_B_method: Union[str, MagneticField_3D_Cartesian] = "torbeam",
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
    auto_delta_sign = True,  # For flipping signs to maintain forward difference. Applies to delta_R and delta_Z
    delta_X: float = -0.0001, # in the same units as data_X_coord
    delta_Y: float = 0.0001,  # in the same units as data_Y_coord
    delta_Z: float = 0.0001,  # in the same units as data_Z_coord
    delta_K_X: float = 0.1,  # in the same units as K_X
    delta_K_Y: float = 0.1,  # in the same units as K_Y
    delta_K_Z: float = 0.1,  # in the same units as K_Z
    interp_order = 5,     # For the 3D interpolation functions
    interp_smoothing = 5, # For the 3D interpolation functions (specifically, density_fit)
    len_tau: int = 102,
    rtol: float = 1e-3,  # for solve_ivp of the beam solver
    atol: float = 1e-6,  # for solve_ivp of the beam solver
    tau_eval: type = None, # To remove
    points_from_2d_scotty_to_eval: ArrayLike = None, # To remove

    # Input and output settings
    ne_data_path = pathlib.Path("."),
    magnetic_data_path = pathlib.Path("."),
    Te_data_path = pathlib.Path("."),
    output_path = pathlib.Path("."),
    input_filename_suffix = "",
    output_filename_suffix = "",
    figure_flag = True,
    detailed_analysis_flag = True,

    # For quick runs (only ray tracing)
    quick_run: bool = False,

    # For launching within the plasma
    plasmaLaunch_K_cartesian = np.zeros(3),
    plasmaLaunch_Psi_3D_lab_cartesian = np.zeros([3, 3]),
    density_fit_method: Optional[Union[str, ProfileFitLike]] = None,
    temperature_fit_method: Optional[Union[str, ProfileFitLike]] = None,

    # For circular flux surfaces
    B_T_axis = None,
    B_p_a = None,
    R_axis = None,
    minor_radius_a = None,
) -> datatree.DataTree:
    
    """
    # TO DO THIS SOON

    # INSERT LONG LIST OF EXPLANATIONS
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

    # Experimental Profile ----------
    if density_fit_parameters is None and (
        density_fit_method in [None, "smoothing-spline-file"]
    ):
        ne_filename = ne_data_path / f"ne{input_filename_suffix}.dat"
        density_fit_parameters = [ne_filename, interp_order, interp_smoothing]

        # FIXME: Read data so it can be saved later
        ne_data = np.fromfile(ne_filename, dtype=float, sep="   ")
        # ne_data_density_array = ne_data[2::2]
        # ne_data_radialcoord_array = ne_data[1::2]
    else:
        ne_filename = None

    find_density_1D = make_fit(
        density_fit_method,
        poloidal_flux_zero_density,
        density_fit_parameters,
        ne_filename,
    )

    """
    # TO DO THIS SOON
    # INSERT RELATIVISTIC_FLAG STUFF
    """
    find_temperature_1D = None
    
    field = create_magnetic_geometry_3D(
        find_B_method,
        magnetic_data_path,
        input_filename_suffix,
        interp_order,
        shot,
        equil_time
    )

    """
    # TO DO THIS SOON
    # Currently this does so in cylindrical -- to convert to cartesian
    # See def poloidal_flux_boundary_along_line (launch.py)

    # Flips the sign of delta_Z depending on the orientation of the poloidal flux surface at the point which the ray enters the plasma.
    # This is to ensure a forward difference across the plasma boundary. We expect poloidal flux to decrease in the direction of the plasma.
    if auto_delta_sign:
        entry_coords = find_entry_point(
            launch_position_cartesian,
            np.deg2rad(poloidal_launch_angle_Torbeam),
            np.deg2rad(toroidal_launch_angle_Torbeam),
            poloidal_flux_enter,
            field,
        )
        entry_X, entry_Y, entry_Z = cylindrical_to_cartesian(entry_coords[0], entry_coords[1], entry_coords[2])

        d_polflux_dX = field.d_polflux_dX(entry_X, entry_Y, entry_Z, delta_X)
        d_polflux_dY = field.d_polflux_dY(entry_X, entry_Y, entry_Z, delta_Y)
        d_polflux_dZ = field.d_polflux_dZ(entry_X, entry_Y, entry_Z, delta_Z)

        if d_polflux_dX > 0: delta_X = -1 * abs(delta_X)
        else: delta_X = abs(delta_X)
        
        if d_polflux_dY > 0: delta_Y = -1 * abs(delta_Y)
        else: delta_Y = abs(delta_Y)
        
        if d_polflux_dZ > 0: delta_Z = -1 * abs(delta_Z)
        else: delta_Z = abs(delta_Z)
    """
    
    # Initialises the Hamiltonian H
    hamiltonian = Hamiltonian_3D(
        field,
        launch_angular_frequency,
        mode_flag,
        find_density_1D,
        delta_X,
        delta_Y,
        delta_Z,
        delta_K_X,
        delta_K_Y,
        delta_K_Z,
        find_temperature_1D
    )

    """
    # TO DO THIS SOON

    # Checking input data
    # Temporarily removed (see check_input.py)
    """

    # ------------------------------
    # Launch parameters #
    # ------------------------------
    if vacuumLaunch_flag:
        pass
        """
        # TO DO THIS SOON

        print("Beam launched from outside the plasma")
        (K_initial_cylindrical,
         initial_position,
         launch_K,
         Psi_3D_lab_initial,
         Psi_3D_lab_launch,
         Psi_3D_lab_entry,
         Psi_3D_lab_entry_cartersian,
         distance_from_launch_to_entry,
        ) = launch_beam(
            toroidal_launch_angle_Torbeam = toroidal_launch_angle_Torbeam,
            poloidal_launch_angle_Torbeam = poloidal_launch_angle_Torbeam,
            launch_beam_width = launch_beam_width,
            launch_beam_curvature = launch_beam_curvature,
            launch_position = launch_position_cylindrical, # because the original function takes in cylindrical coordinates only
            launch_angular_frequency = launch_angular_frequency,
            mode_flag = mode_flag,
            field = field,
            hamiltonian = hamiltonian,
            vacuum_propagation_flag = vacuum_propagation_flag,
            Psi_BC_flag = Psi_BC_flag,
            poloidal_flux_enter = poloidal_flux_enter,
            delta_R = delta_X,
            delta_Z = delta_Z,
        )
        """
    else:
        print("Beam launched from inside the plasma")
        Psi_3D_lab_initial_cartesian = plasmaLaunch_Psi_3D_lab_cartesian
        K_initial_cartesian = plasmaLaunch_K_cartesian
        initial_position = launch_position_cartesian
        launch_K = None
        Psi_3D_lab_launch_cartesian = None
        Psi_3D_lab_entry_cartersian = None
        distance_from_launch_to_entry = None

    # -------------------
    # Propagate the ray

    if points_from_2d_scotty_to_eval is None: # TO REMOVE this entire if-else block
        print("Starting the solvers")
        ray_solver_output = propagate_ray(
            poloidal_flux_enter,
            launch_angular_frequency,
            field,
            initial_position,
            K_initial_cartesian,
            hamiltonian,
            rtol,
            atol,
            quick_run,
            len_tau,
            tau_eval, # TO REMOVE
        )
        if quick_run:
            return ray_solver_output

        tau_leave, tau_points = cast(tuple, ray_solver_output)

        # return ray_solver_output # TO REMOVE -- for debugging purposes only

        # -------------------
        # Propagate the beam

        # Initial conditions for the solver
        beam_parameters_initial = pack_beam_parameters_3D(
            initial_position[0],
            initial_position[1],
            initial_position[2],
            K_initial_cartesian[0],
            K_initial_cartesian[1],
            K_initial_cartesian[2],
            Psi_3D_lab_initial_cartesian,
        )

        print("Psi: ") # TO REMOVE
        print("Psi_xx: ", beam_parameters_initial[6], beam_parameters_initial[12])
        print("Psi_yy: ", beam_parameters_initial[7], beam_parameters_initial[13])
        print("Psi_zz: ", beam_parameters_initial[8], beam_parameters_initial[14])

        print("Psi_xy: ", beam_parameters_initial[9], beam_parameters_initial[15])
        print("Psi_xz: ", beam_parameters_initial[10], beam_parameters_initial[16])
        print("Psi_yz: ", beam_parameters_initial[11], beam_parameters_initial[17])

        solver_start_time = time.time()

        # TO REMOVE
        tau_points = list(tau_points)
        tau_points = dict.fromkeys(tau_points)
        tau_points = list(tau_points)
        tau_points = np.array(tau_points)
        print(tau_points)

        solver_beam_output = solve_ivp(
            beam_evolution_fun_3D,
            [0, tau_leave],
            beam_parameters_initial,
            method="RK45",
            t_eval=tau_points,
            dense_output=False,
            events=None,
            vectorized=False,
            args=(hamiltonian,),
            rtol=rtol,
            atol=atol,
        )

        solver_end_time = time.time()
        solver_time = solver_end_time - solver_start_time
        print(f"Time taken (beam solver) {solver_time}s")
        print(f"Number of beam evolution evaluations: {solver_beam_output.nfev}")
        print(f"Time per beam evolution evaluation: {solver_time / solver_beam_output.nfev}")

        tau_array = solver_beam_output.t
        beam_parameters_final = solver_beam_output.y
        solver_status = solver_beam_output.status

        q_X_array, q_Y_array, q_Z_array, K_X_array, K_Y_array, K_Z_array, Psi_3D_output = unpack_beam_parameters_3D(beam_parameters_final)

        print("Main loop complete")
        # -------------------
    else:
        solver_status = 1
        q_X_array     = points_from_2d_scotty_to_eval[0]
        q_Y_array     = points_from_2d_scotty_to_eval[1]
        q_Z_array     = points_from_2d_scotty_to_eval[2]
        K_X_array     = points_from_2d_scotty_to_eval[3]
        K_Y_array     = points_from_2d_scotty_to_eval[4]
        K_Z_array     = points_from_2d_scotty_to_eval[5]
        Psi_3D_output = points_from_2d_scotty_to_eval[6]
        tau_array     = points_from_2d_scotty_to_eval[7]

        polflux_values = points_from_2d_scotty_to_eval[8]
        theta_m_array  = points_from_2d_scotty_to_eval[9]



    # TO REMOVE
    if solver_status == -1: print("Solver did not reach completion.")
    else:
        pointwise_data = [[q_X_array[i],
                           q_Y_array[i],
                           q_Z_array[i],
                           K_X_array[i],
                           K_Y_array[i],
                           K_Z_array[i],
                           Psi_3D_output[i],
                           tau_array[i]]
                           for i in range(len(tau_array))]
        
        ### THIS IS TO SEE HOW POLFLUX, ELECTRON DENSITY, AND H CHANGE W.R.T. TAU
        # to remove
        if points_from_2d_scotty_to_eval is None or polflux_values is None: polflux_values = field.polflux(q_X_array, q_Y_array, q_Z_array)
        ne_values = find_density_1D(polflux_values)
        H_values = hamiltonian(q_X_array, q_Y_array, q_Z_array, K_X_array, K_Y_array, K_Z_array)

        ### THIS IS TO SEE HOW B_MAGNITUDE, K_MAGNITUDE, THETA_M, THE EPSILONS, AND THE BOOKER TERMS CHANGE W.R.T TAU
        # to remove
        from scotty.fun_general import dot, angular_frequency_to_wavenumber
        from scotty.hamiltonian_3D import DielectricTensor_3D
        _B_X, _B_Y, _B_Z = np.squeeze(field.B_X(q_X_array, q_Y_array, q_Z_array)), np.squeeze(field.B_Y(q_X_array, q_Y_array, q_Z_array)), np.squeeze(field.B_Z(q_X_array, q_Y_array, q_Z_array))
        B_magnitude = np.sqrt(_B_X**2 + _B_Y**2 + _B_Z**2)
        _b_hat = np.array([_B_X, _B_Y, _B_Z]) / B_magnitude
        _K_X, _K_Y, _K_Z = np.array(K_X_array), np.array(K_Y_array), np.array(K_Z_array)
        K_magnitude = np.sqrt(_K_X**2 + _K_Y**2 + _K_Z**2)
        _K_hat = np.array([_K_X, _K_Y, _K_Z]) / K_magnitude
        if points_from_2d_scotty_to_eval is None or theta_m_array is None:
            theta_m = np.arcsin(dot(_b_hat.T, _K_hat.T))
            _sin_theta_m_sq = dot(_b_hat.T, _K_hat.T)**2
        else:
            theta_m = theta_m_array
            _sin_theta_m_sq = np.sin(theta_m)**2
        _epsilon = DielectricTensor_3D(ne_values, launch_angular_frequency, B_magnitude)
        epsilon_para = _epsilon.e_bb
        epsilon_perp = _epsilon.e_11
        epsilon_g    = _epsilon.e_12
        Booker_alpha = (_epsilon.e_bb * _sin_theta_m_sq) + _epsilon.e_11 * (1 - _sin_theta_m_sq)
        Booker_beta  = (-_epsilon.e_11 * _epsilon.e_bb * (1 + _sin_theta_m_sq)) - (_epsilon.e_11**2 - _epsilon.e_12**2) * (1 - _sin_theta_m_sq)
        Booker_gamma = _epsilon.e_bb * (_epsilon.e_11**2 - _epsilon.e_12**2)
        H_discriminant = np.maximum(np.zeros_like(Booker_beta), (Booker_beta**2 - 4 * Booker_alpha * Booker_gamma))
        H_first_term  = (K_magnitude / angular_frequency_to_wavenumber(launch_angular_frequency))**2
        H_second_term = (Booker_beta - (-1)*np.sqrt(H_discriminant)) / (2*Booker_alpha)
        H_first_second_term = H_first_term + H_second_term

        ### THIS IS TO SEE HOW THE FIRST AND SECOND DERIVATIVES CHANGE W.R.T. TAU
        # to remove
        first_order_derivatives_dict = {
            "dH_dX": [],
            "dH_dY": [],
            "dH_dZ": [],
            "dH_dKx": [],
            "dH_dKy": [],
            "dH_dKz": []
        }
        second_order_derivatives_dict = {
            "d2H_dX2": [],
            "d2H_dY2": [],
            "d2H_dZ2": [],
            "d2H_dX_dY": [],
            "d2H_dX_dZ": [],
            "d2H_dY_dZ": [],
            "d2H_dKx2": [],
            "d2H_dKy2": [],
            "d2H_dKz2": [],
            "d2H_dKx_dKy": [],
            "d2H_dKx_dKz": [],
            "d2H_dKy_dKz": [],
            "d2H_dX_dKx": [],
            "d2H_dX_dKy": [],
            "d2H_dX_dKz": [],
            "d2H_dY_dKx": [],
            "d2H_dY_dKy": [],
            "d2H_dY_dKz": [],
            "d2H_dZ_dKx": [],
            "d2H_dZ_dKy": [],
            "d2H_dZ_dKz": [],
        }
    
        # to remove
        from scotty.hamiltonian_3D import hessians_3D
        def calculate_and_store_derivatives(index):
            X, Y, Z, K_X, K_Y, K_Z, Psi_3D, tau = pointwise_data[index]
            derivatives_and_second_derivatives_dict = hamiltonian.derivatives(X, Y, Z, K_X, K_Y, K_Z, second_order = True)

            grad_grad_H, gradK_grad_H, gradK_gradK_H = hessians_3D(derivatives_and_second_derivatives_dict)
            gradK_grad_H = np.transpose(gradK_grad_H)

            """
            print()
            print("grad_grad_H should = 0")
            print(grad_grad_H)
            print()
            print("grad_gradK_H should = 0")
            print(grad_gradK_H)
            print()
            print("gradK_grad_H should = 0")
            print(gradK_grad_H)
            print()
            print("gradK_gradK_H should = 0")
            print(gradK_gradK_H)
            print()
            print()
            print()
            """

            for key, value in derivatives_and_second_derivatives_dict.items():
                if key in ["dH_dX", "dH_dY", "dH_dZ", "dH_dKx", "dH_dKy", "dH_dKz"]: first_order_derivatives_dict[key].append(value)
                else: second_order_derivatives_dict[key].append(value)

        # to remove
        for index in range(len(tau_array)):
            calculate_and_store_derivatives(index)
        
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MultipleLocator
        ### THIS IS TO SEE HOW THE FIRST DERIVATIVES CHANGE W.R.T. TAU
        def plot_how_first_derivatives_evolve(dictionary):
            counter = 0
            for key, value in dictionary.items():
                # xmin, xmax = 0, 900
                # ymin, ymax = min(value[xmin:xmax+1]), max(value[xmin:xmax+1])
                # axs[counter].set_xlim(xmin, xmax)
                # axs[counter].set_ylim(ymin, ymax)
                axs[counter].plot(tau_array, value) #, marker='o')
                axs[counter].set_xlabel("tau")
                axs[counter].set_ylabel(key)
                axs[counter].set_title(f"{key} changing along tau")
                axs[counter].xaxis.set_major_locator(MultipleLocator(75))
                # axs[counter].xaxis.set_minor_locator(MultipleLocator(25))
                axs[counter].grid(True, which="both", axis="x")
                counter += 1

        fig, axs = plt.subplots(2, 3, figsize=(15, 5))
        axs = axs.flatten()
        plot_how_first_derivatives_evolve(first_order_derivatives_dict)
        plt.tight_layout()
        plt.show()

        # THIS IS TO SEE HOW THE SECOND DERIVATIVES CHANGE W.R.T. TAU
        def plot_how_second_derivatives_evolve(dictionary):
            counter = 0
            for key, value in dictionary.items():
                # xmin, xmax = 0, 1002 # 0, 900
                # ymin, ymax = min(value[xmin:xmax+1]), max(value[xmin:xmax+1])
                # axs[counter].set_xlim(xmin, xmax)
                # axs[counter].set_ylim(ymin, ymax)
                axs[counter].plot(tau_array, value) #, marker='o')
                axs[counter].set_xlabel("tau")
                axs[counter].set_ylabel(key)
                axs[counter].set_title(f"{key} changing along tau")
                axs[counter].xaxis.set_major_locator(MultipleLocator(75))
                # axs[counter].xaxis.set_minor_locator(MultipleLocator(25))
                axs[counter].grid(True, which="both", axis="x")
                counter += 1

        fig, axs = plt.subplots(7, 3, figsize=(15, 15))
        axs = axs.flatten()
        plot_how_second_derivatives_evolve(second_order_derivatives_dict)
        plt.tight_layout()
        plt.show()

        pointwise_data = [[q_X_array[i],
                           q_Y_array[i],
                           q_Z_array[i],
                           K_X_array[i],
                           K_Y_array[i],
                           K_Z_array[i],
                           Psi_3D_output[i],
                           tau_array[i],
                           first_order_derivatives_dict["dH_dX"][i],
                           first_order_derivatives_dict["dH_dY"][i],
                           first_order_derivatives_dict["dH_dZ"][i],
                           first_order_derivatives_dict["dH_dKx"][i],
                           first_order_derivatives_dict["dH_dKy"][i],
                           first_order_derivatives_dict["dH_dKz"][i],
                           second_order_derivatives_dict["d2H_dX2"][i],
                           second_order_derivatives_dict["d2H_dY2"][i],
                           second_order_derivatives_dict["d2H_dZ2"][i],
                           second_order_derivatives_dict["d2H_dX_dY"][i],
                           second_order_derivatives_dict["d2H_dX_dZ"][i],
                           second_order_derivatives_dict["d2H_dY_dZ"][i],
                           second_order_derivatives_dict["d2H_dKx2"][i],
                           second_order_derivatives_dict["d2H_dKy2"][i],
                           second_order_derivatives_dict["d2H_dKz2"][i],
                           second_order_derivatives_dict["d2H_dKx_dKy"][i],
                           second_order_derivatives_dict["d2H_dKx_dKz"][i],
                           second_order_derivatives_dict["d2H_dKy_dKz"][i],
                           second_order_derivatives_dict["d2H_dX_dKx"][i],
                           second_order_derivatives_dict["d2H_dX_dKy"][i],
                           second_order_derivatives_dict["d2H_dX_dKz"][i],
                           second_order_derivatives_dict["d2H_dY_dKx"][i],
                           second_order_derivatives_dict["d2H_dY_dKy"][i],
                           second_order_derivatives_dict["d2H_dY_dKz"][i],
                           second_order_derivatives_dict["d2H_dZ_dKx"][i],
                           second_order_derivatives_dict["d2H_dZ_dKy"][i],
                           second_order_derivatives_dict["d2H_dZ_dKz"][i],
                           H_values[i],
                           polflux_values[i],
                           ne_values[i],
                           B_magnitude[i],
                           K_magnitude[i],
                           theta_m[i],
                           epsilon_para[i],
                           epsilon_perp[i],
                           epsilon_g[i],
                           Booker_alpha[i],
                           Booker_beta[i],
                           Booker_gamma[i],
                           H_discriminant[i],
                           H_first_term[i],
                           H_second_term[i],
                           H_first_second_term[i]]
                           for i in range(len(tau_array))]
        
        # TO REMOVE
        # This is to analytically calculate H as a function of Kz and tau
        # specifically only for the 2d slab geometry, for which we know
        # the analytical expression for H
        _Kz_array_for_H = np.linspace(start=-1, stop=1, num=201)
        _numerical_H_values_for_heatmap = []
        for i in range(len(pointwise_data)):
            _H_values_row = []
            for Kz in _Kz_array_for_H:
                _H_values_single_point = hamiltonian(q_X_array[i], q_Y_array[i], q_Z_array[i], K_X_array[i], K_Y_array[i], Kz)
                _H_values_row.append(_H_values_single_point)
            _H_values_row = np.array(_H_values_row)
            _numerical_H_values_for_heatmap.append(_H_values_row)
        _numerical_H_values_for_heatmap = np.array(_numerical_H_values_for_heatmap)
    
    return pointwise_data, _numerical_H_values_for_heatmap # TO REMOVE










    """
    inputs = xr.Dataset(
        {
            "B_T_axis": B_T_axis,
            "B_p_a": B_p_a,
            "K_initial": (["col"], K_initial_cartesian),
            "Psi_BC_flag": Psi_BC_flag,
            "R_axis": R_axis,
            "atol": atol,
            "delta_X": delta_X,
            "delta_Y": delta_Y,
            "delta_Z": delta_Z,
            "delta_K_X": delta_K_X,
            "delta_K_Y": delta_K_Y,
            "delta_K_Z": delta_K_Z,
            "density_fit_method": str(density_fit_method),
            "density_fit_parameters": str(density_fit_parameters),
            "detailed_analysis_flag": detailed_analysis_flag,
            "equil_time": (equil_time),
            "figure_flag": figure_flag,
            "find_B_method": str(find_B_method),
            "initial_position": (["col"], initial_position),
            "input_filename_suffix": input_filename_suffix,
            "interp_order": interp_order,
            # "interp_smoothing": interp_smoothing,
            "launch_K": (launch_K),
            "launch_angular_frequency": launch_angular_frequency,
            "launch_beam_curvature": launch_beam_curvature,
            "launch_beam_width": launch_beam_width,
            "launch_freq_GHz": launch_freq_GHz,
            "launch_position": (["col"], launch_position_cartesian),
            "len_tau": len_tau,
            "magnetic_data_path": str(magnetic_data_path),
            "minor_radius_a": minor_radius_a,
            "mode_flag": mode_flag,
            "ne_data_density_array": (ne_data_density_array),
            "ne_data_path": str(ne_data_path),
            "ne_data_radialcoord_array": (ne_data_radialcoord_array),
            "output_filename_suffix": output_filename_suffix,
            "output_path": str(output_path),
            "plasmaLaunch_K": plasmaLaunch_K_cartesian,
            "plasmaLaunch_Psi_3D_lab_Cartesian": (
                ["row", "col"],
                plasmaLaunch_Psi_3D_lab_cartesian,
            ),
            "poloidalFlux_grid": (["R", "Z"], field.polflux_grid),
            "poloidal_flux_enter": poloidal_flux_enter,
            "poloidal_launch_angle_Torbeam": poloidal_launch_angle_Torbeam,
            "Psi_3D_lab_initial": (
                ["row", "col"],
                Psi_3D_lab_initial_cartesian,
            ),
            "quick_run": quick_run,
            "rtol": rtol,
            "shot": shot,
            "toroidal_launch_angle_Torbeam": toroidal_launch_angle_Torbeam,
            "vacuumLaunch_flag": vacuumLaunch_flag,
            "vacuum_propagation_flag": vacuum_propagation_flag,
        },
        coords = {
            "X": field.X_coord,
            "Y": field.Y_coord,
            "Z": field.Z_coord,
            "row": ["X", "Y", "Z"],
            "col": ["X", "Y", "Z"],
        },
    )

    solver_output = xr.Dataset(
        {
            "solver_status": solver_status,
            "q_X": (["tau"], q_X_array, {"long_name": "X", "units": "m"}),
            "q_Y": (["tau"], q_Y_array, {"long_name": "Y", "units": "m"}),
            "q_Z": (["tau"], q_Z_array, {"long_name": "Z", "units": "m"}),

            "K_X": (["tau"], K_X_array),
            "K_Y": (["tau"], K_Y_array),
            "K_Z": (["tau"], K_Z_array),
            "Psi_3D": (["tau", "row", "col"], Psi_3D_output)
        },
        coords = {
            "tau": tau_array,
            "row": ["X", "Y", "Z"],
            "col": ["X", "Y", "Z"],
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
    dH = hamiltonian.derivatives(q_X_array, q_Y_array, q_Z_array, K_X_array, K_Y_array, K_Z_array, second_order = True)

    # df = immediate_analysis
    """










def make_fit(
    method: Optional[Union[str, ProfileFitLike]],
    poloidal_flux_zero_density: float,
    parameters: Optional[Sequence],
    filename: Optional[PathLike]
) -> ProfileFitLike:
    
    if callable(method):
        return method
    
    if not isinstance(method, (str, type(None))):
        raise TypeError(
            f"Unexpected method type. Expected callable, str, or None, but got '{type(method)}'"
        )
    
    if parameters is None:
        raise ValueError(
            f"Passing `density_fit_method` ({method}) as string or None requires a list or array of parameters"
        )
    
    return profile_fit(method, poloidal_flux_zero_density, parameters, filename)





def create_magnetic_geometry_3D(
    find_B_method: Union[str, MagneticField_3D_Cartesian],
    magnetic_data_path: Optional[pathlib.Path] = None,
    input_filename_suffix: str = "",
    interp_order: int = 5,
    shot: Optional[int] = None,
    equil_time: Optional[float] = None,
    **kwargs
) -> MagneticField_3D_Cartesian:
    
    if isinstance(find_B_method, MagneticField_3D_Cartesian): return find_B_method

    def missing_arg(argument: str) -> str:
        return f"Missing '{argument}' for find_B_method='{find_B_method}'"
    
    if find_B_method == "omfit_3D":
        print("Using OMFIT JSON Torbeam file for B and polflux")
        topfile_filename = magnetic_data_path / f"topfile{input_filename_suffix}.json"

        with open(topfile_filename) as f:
            data = json.load(f)
        
        X_grid = np.array(data["X"])
        Y_grid = np.array(data["Y"])
        Z_grid = np.array(data["Z"])
        B_X = np.array(data["Bx"])
        B_Y = np.array(data["By"])
        B_Z = np.array(data["Bz"])
        polflux = np.array(data["pol_flux"])

        return InterpolatedField_3D_Cartesian(
            X_grid,
            Y_grid,
            Z_grid,
            B_X,
            B_Y,
            B_Z,
            polflux,
            interp_order)