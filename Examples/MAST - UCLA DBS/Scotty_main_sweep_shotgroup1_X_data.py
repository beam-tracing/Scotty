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

from scotty.init_bruv import get_parameters_for_Scotty


# equil_times = np.linspace(0.16,0.25,10)
# polflux_enter = np.array([1.18662487,1.16317467,1.16060602,
#                           1.17713456,1.19524207,1.15081960,
#                           1.15031088,1.13624193,1.16034804,
#                           1.14965615])**2
equil_times = np.linspace(0.23, 0.25, 3)
polflux_enter = np.array([1.13624193, 1.16034804, 1.14965615]) ** 2

mirror_rotations = np.linspace(3, -6, 46)
# mirror_rotations = np.linspace(3,1.2,10)
mirror_tilt = -4
launch_freqs_GHz = np.array([30.0, 32.5, 35.0, 37.5, 42.5, 45.0, 47.5, 50.0])


for ii, equil_time in enumerate(equil_times):
    for jj, mirror_rotation in enumerate(mirror_rotations):
        for kk, launch_freq_GHz in enumerate(launch_freqs_GHz):
            kwargs_dict = get_parameters_for_Scotty(
                "DBS_NSTX_MAST",
                launch_freq_GHz=launch_freq_GHz,
                mirror_rotation=mirror_rotation,  # angle, in deg
                mirror_tilt=mirror_tilt,  # angle, in deg
                find_B_method="EFITpp",  # EFITpp, UDA_saved, UDA, torbeam
                find_ne_method="poly3",
                equil_time=equil_time,
                shot=29908,
                user="Valerian_laptop",
            )

            if kwargs_dict["launch_freq_GHz"] > 52.5:
                kwargs_dict["mode_flag"] = 1
            else:
                kwargs_dict["mode_flag"] = -1

            if kwargs_dict["mode_flag"] == 1:
                mode_string = "O"
            elif kwargs_dict["mode_flag"] == -1:
                mode_string = "X"

            kwargs_dict["output_filename_suffix"] = (
                "_r" + f"{mirror_rotation:.1f}"
                "_t"
                + f"{mirror_tilt:.1f}"
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
            ] = "C:\\Dropbox\\VHChen2021\\Data - Scotty\\Run 24\\"
            kwargs_dict["density_fit_parameters"] = None
            kwargs_dict[
                "ne_data_path"
            ] = "C:\\Dropbox\\VHChen2021\\Data - Equilibrium\MAST\\"
            kwargs_dict["input_filename_suffix"] = (
                "_shotgroup1_avr_" + f"{equil_time*1000:.0f}" + "ms"
            )

            kwargs_dict["poloidal_flux_enter"] = polflux_enter[ii]

            kwargs_dict["delta_R"] = -0.0001
            kwargs_dict["delta_Z"] = -0.0001
            kwargs_dict["delta_K_R"] = 0.1
            kwargs_dict["delta_K_zeta"] = 0.1
            kwargs_dict["delta_K_Z"] = 0.1
            kwargs_dict["interp_smoothing"] = 0.0
            kwargs_dict["len_tau"] = 1002
            kwargs_dict["rtol"] = 1e-3
            kwargs_dict["atol"] = 1e-6

            if ii == 0 and jj == 0 and kk == 0:
                kwargs_dict["verbose_output_flag"] = True
            else:
                kwargs_dict["verbose_output_flag"] = False

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
            # if os.path.exists(data_output) and os.path.exists(analysis_output):
            #     continue
            # else:
            #     beam_me_up(**kwargs_dict)

            beam_me_up(**kwargs_dict)
