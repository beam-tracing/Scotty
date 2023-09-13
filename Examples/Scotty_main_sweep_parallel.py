# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 10:44:34 2018

@author: VH Hall-Chen
Valerian Hongjie Hall-Chen
valerian@hall-chen.com


For shot 29908, the EFIT++ times are efit_times = np.linspace(0.155,0.25,20)
I want efit_times[np.arange(0,10)*2 + 1]. 160ms, 170ms, ..., 250ms
"""
from scotty.beam_me_up import beam_me_up
import numpy as np
import os
from joblib import Parallel, delayed

from scotty.init_bruv import get_parameters_for_Scotty


# poloidal_launch_angles_Torbeam = np.linspace(2.0,6.0,5)
poloidal_launch_angles_Torbeam = np.linspace(4.0, 15.0, 12)
toroidal_launch_angles_Torbeam = np.linspace(3.0, 10.0, 8)
launch_freqs_GHz = np.array([32.5, 50.0])


def loop(
    poloidal_launch_angle_Torbeam,
    toroidal_launch_angles_Torbeam=toroidal_launch_angles_Torbeam,
    launch_freqs_GHz=launch_freqs_GHz,
):

    shot = 45118
    equil_time = 0.6

    for kk, mode_flag in enumerate([1, -1]):
        for ii, toroidal_launch_angle_Torbeam in enumerate(
            toroidal_launch_angles_Torbeam
        ):
            for jj, launch_freq_GHz in enumerate(launch_freqs_GHz):
                kwargs_dict = get_parameters_for_Scotty(
                    "DBS_UCLA_MAST-U",
                    launch_freq_GHz=launch_freq_GHz,
                    find_B_method="test",  # EFITpp, UDA_saved, UDA, torbeam
                    find_ne_method=None,
                    equil_time=equil_time,
                    shot=shot,
                    user="Valerian_laptop",
                )

                kwargs_dict["mode_flag"] = mode_flag
                kwargs_dict[
                    "poloidal_launch_angle_Torbeam"
                ] = poloidal_launch_angle_Torbeam
                kwargs_dict[
                    "toroidal_launch_angle_Torbeam"
                ] = toroidal_launch_angle_Torbeam

                if kwargs_dict["mode_flag"] == 1:
                    mode_string = "O"
                elif kwargs_dict["mode_flag"] == -1:
                    mode_string = "X"

                kwargs_dict["output_filename_suffix"] = (
                    "_pol" + f"{poloidal_launch_angle_Torbeam:.1f}"
                    "_tor"
                    + f"{toroidal_launch_angle_Torbeam:.1f}"
                    + "_f"
                    + f"{launch_freq_GHz:.1f}"
                    + "_"
                    + mode_string
                    + "_"
                    + f"{equil_time*1000:.3g}"
                    + "ms"
                )

                kwargs_dict["figure_flag"] = False
                kwargs_dict[
                    "output_path"
                ] = "C:\\Dropbox\\VHChen2021\\Data - Scotty\\Run 26\\"
                kwargs_dict["density_fit_parameters"] = None
                kwargs_dict[
                    "ne_data_path"
                ] = "C:\\Dropbox\\VHChen2021\\Data - Equilibrium\MAST-U\\"
                kwargs_dict[
                    "magnetic_data_path"
                ] = "C:\\Dropbox\\VHChen2021\\Data - Equilibrium\MAST-U\\"
                kwargs_dict["input_filename_suffix"] = (
                    "_" + str(shot) + "_" + f"{equil_time*1000:.0f}" + "ms"
                )
                kwargs_dict["detailed_analysis_flag"] = False

                kwargs_dict["poloidal_flux_enter"] = 1.04080884**2
                kwargs_dict["Psi_BC_flag"] = True
                kwargs_dict["vacuum_propagation_flag"] = True

                kwargs_dict["delta_R"] = -0.0001
                kwargs_dict["delta_Z"] = 0.0001
                kwargs_dict["delta_K_R"] = 0.1
                kwargs_dict["delta_K_zeta"] = 0.1
                kwargs_dict["delta_K_Z"] = 0.1
                kwargs_dict["interp_smoothing"] = 0.0
                kwargs_dict["len_tau"] = 1002
                kwargs_dict["rtol"] = 1e-3
                kwargs_dict["atol"] = 1e-6

                data_output = (
                    kwargs_dict["output_path"]
                    + "data_output"
                    + kwargs_dict["output_filename_suffix"]
                    + ".npz"
                )
                analysis_output = (
                    kwargs_dict["output_path"]
                    + "analysis_output"
                    + kwargs_dict["output_filename_suffix"]
                    + ".npz"
                )
                if os.path.exists(data_output) and os.path.exists(analysis_output):
                    continue
                else:
                    beam_me_up(**kwargs_dict)
                return


Parallel(n_jobs=3)(
    delayed(loop)(poloidal_launch_angle_Torbeam)
    for poloidal_launch_angle_Torbeam in poloidal_launch_angles_Torbeam
)
