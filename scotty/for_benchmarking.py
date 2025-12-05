from copy import deepcopy
import numpy as np
from scotty.benchmarking_3D import benchmark_me_up_3D, benchmark_plots
from scotty.fun_general import cylindrical_to_cartesian

def get_launch_stuff(
        DBS_system = "BEST",
        mode_to_try = "O",
        launch_setting = 1):

    if DBS_system == "DIII-D":
        launch_position_cylindrical = np.array([3.01346, 0, -0.09017])
        launch_position_cartesian = launch_position_cylindrical
        poloidal_launch_angle_Torbeam = -7.0
        toroidal_launch_angle_Torbeam = 0 + np.rad2deg(np.arctan2(launch_position_cartesian[1], launch_position_cartesian[0]))
        mode_flag = mode_to_try
        launch_freq_GHz = 72.5
        launch_beam_width = 0.1265
        launch_beam_curvature = -0.95
        Psi_BC_flag = "discontinuous"
        poloidal_flux_enter = 0.95
        poloidal_flux_zero_density = poloidal_flux_enter + 0.01
        file_dir = r"C:\Users\eduar\OneDrive\OneDrive - EWIKARTA001\OneDrive - Nanyang Technological University\Work\Scotty Benchmarks\DIII-D shot 189998\cyl scotty"
        file_suffix = "_189998_3000ms_quinn"

    elif DBS_system == "MAST-U":
        launch_position_cylindrical = np.array([2.278, 0, 0])
        launch_position_cartesian = launch_position_cylindrical
        poloidal_launch_angle_Torbeam = -8.2
        mode_flag = mode_to_try
        Psi_BC_flag = "discontinuous"
        file_dir = r"C:\Users\eduar\OneDrive\OneDrive - EWIKARTA001\OneDrive - Nanyang Technological University\Work\Scotty Benchmarks\MAST-U\cyl scotty"
        file_suffix = "_51353_410ms_1.0"

        if launch_setting == 1:
            poloidal_flux_enter = 1.09**2
            toroidal_launch_angle_Torbeam = -10.1 + np.rad2deg(np.arctan2(launch_position_cartesian[1], launch_position_cartesian[0])) # 7.10 + np.rad2deg(np.arctan2(launch_position_cartesian[1], launch_position_cartesian[0]))
            launch_freq_GHz = 32.5
            launch_beam_width = 0.07596928872724663
            launch_beam_curvature = -0.7497156475519201
        elif launch_setting == 2:
            poloidal_flux_enter = 1.09**2
            toroidal_launch_angle_Torbeam = -10.1 + np.rad2deg(np.arctan2(launch_position_cartesian[1], launch_position_cartesian[0])) # 7.10 + np.rad2deg(np.arctan2(launch_position_cartesian[1], launch_position_cartesian[0]))
            launch_freq_GHz = 50.1
            launch_beam_width = 0.04952158283198232
            launch_beam_curvature = -0.8079274424969226
        
        poloidal_flux_zero_density = poloidal_flux_enter + 0.01

    return {
        "poloidal_launch_angle_Torbeam": poloidal_launch_angle_Torbeam,
        "toroidal_launch_angle_Torbeam": toroidal_launch_angle_Torbeam,
        "launch_freq_GHz": launch_freq_GHz,
        "launch_beam_width": launch_beam_width,
        "launch_beam_curvature": launch_beam_curvature,
        "launch_position": launch_position_cylindrical,
        "mode_flag_cyl": 1 if mode_flag in [1, "O"] else -1,
        "mode_flag_cart": mode_flag,
        "vacuumLaunch_flag": True,
        "vacuum_propagation_flag": True,
        "Psi_BC_flag": Psi_BC_flag,
        "poloidal_flux_enter": poloidal_flux_enter,
        "poloidal_flux_zero_density": poloidal_flux_zero_density,
        "ne_data_path": file_dir,
        "magnetic_data_path": file_dir,
        "input_filename_suffix": file_suffix,
        "output_path": file_dir,
    }

# cyl scotty kwargs

kwargs_dict_cyl = get_launch_stuff(
    DBS_system = "DIII-D",
    mode_to_try = "O",
    launch_setting = 1)

kwargs_dict_cart = deepcopy(kwargs_dict_cyl)
kwargs_dict_cyl["mode_flag"] = kwargs_dict_cyl.pop("mode_flag_cyl")
kwargs_dict_cyl.pop("mode_flag_cart")
kwargs_dict_cyl["delta_R"] = -1e-3
kwargs_dict_cyl["delta_Z"] = 1e-3
kwargs_dict_cyl["delta_K_R"]    = -1e-1
kwargs_dict_cyl["delta_K_zeta"] = 1e-1
kwargs_dict_cyl["delta_K_Z"]    = 1e-1
kwargs_dict_cyl["interp_smoothing"] = 0
kwargs_dict_cyl["interp_order"] = 5
kwargs_dict_cyl["len_tau"] = 1002
kwargs_dict_cyl["rtol"] = 1e-4
kwargs_dict_cyl["atol"] = 1e-7
kwargs_dict_cyl["find_B_method"] = "omfit"

# cart scotty kwargs

kwargs_dict_cart["mode_flag"] = kwargs_dict_cart.pop("mode_flag_cart")
kwargs_dict_cart.pop("mode_flag_cyl")
kwargs_dict_cart["launch_position_cartesian"] = kwargs_dict_cart.pop("launch_position")
kwargs_dict_cart["auto_delta_sign"] = True
kwargs_dict_cart["delta_X"] = -1e-3
kwargs_dict_cart["delta_Y"] = 1e-3
kwargs_dict_cart["delta_Z"] = 1e-3
kwargs_dict_cart["delta_K_X"] = -1e-1
kwargs_dict_cart["delta_K_Y"] = 1e-1
kwargs_dict_cart["delta_K_Z"] = 1e-1
kwargs_dict_cart["interp_smoothing"] = 0
kwargs_dict_cart["interp_order"] = 5
kwargs_dict_cart["len_tau"] = 1002
kwargs_dict_cart["rtol"] = 1e-4
kwargs_dict_cart["atol"] = 1e-7
kwargs_dict_cart["further_analysis_flag"] = True
kwargs_dict_cart["figure_flag"] = True
kwargs_dict_cart["return_dt_field"] = True
kwargs_dict_cart["console_log_level"] = "debug"
kwargs_dict_cart["file_log_level"] = "debug"
kwargs_dict_cart["create_magnetic_geometry_3D"] = {
    "pad": 35,
    "Y_spacing": 0.02,
}

# benchmarking

cyl, cart = benchmark_me_up_3D(kwargs_dict_cyl, kwargs_dict_cart)

(dt_cyl_scotty,
 field_cyl_scotty,
 ne_cyl_scotty,
 temeprature_cyl_scotty,
 H_cyl_scotty) = cyl

(dt_cart_scotty,
 field_cart_scotty,
 ne_cart_scotty,
 temeprature_cart_scotty,
 H_cart_scotty) = cart

# printing plots

benchmark_plots(dt_cyl_scotty, dt_cart_scotty)