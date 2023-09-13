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


equil_times = np.linspace(0.16, 0.25, 10)
# mirror_rotations = np.linspace(1,-6,36)
mirror_rotations = np.linspace(1.2, 3, 10)
mirror_tilt = -4
launch_freqs_GHz = np.array(
    [
        30.0,
        32.5,
        35.0,
        37.5,
        42.5,
        45.0,
        47.5,
        50.0,
        55.0,
        57.5,
        60.0,
        62.5,
        67.5,
        70.0,
        72.5,
        75.0,
    ]
)

counter = 0
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
                user="Valerian_desktop",
            )

            if kwargs_dict["launch_freq_GHz"] > 52.5:
                kwargs_dict["mode_flag"] = 1
            else:
                kwargs_dict["mode_flag"] = -1

            if kwargs_dict["mode_flag"] == 1:
                mode_string = "O"
            elif kwargs_dict["mode_flag"] == -1:
                mode_string = "X"

            if counter == 0:
                kwargs_dict["verbose_output_flag"] = True
            else:
                kwargs_dict["verbose_output_flag"] = False

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
            kwargs_dict["detailed_analysis_flag"] = False
            kwargs_dict[
                "output_path"
            ] = "D:\\Dropbox\\VHChen2021\\Data - Scotty\\Run 18\\"

            # an_output = kwargs_dict['output_path'] + 'data_output' + kwargs_dict['output_filename_suffix'] + '.npz'
            # if os.path.exists(an_output):
            #     continue
            # else:
            #     beam_me_up(**kwargs_dict)

            kwargs_dict["delta_R"] = -0.00001
            kwargs_dict["delta_Z"] = -0.00001
            kwargs_dict["delta_K_R"] = 0.1
            kwargs_dict["delta_K_zeta"] = 0.1
            kwargs_dict["delta_K_Z"] = 0.1
            kwargs_dict["interp_smoothing"] = 2.0
            kwargs_dict["len_tau"] = 102
            kwargs_dict["rtol"] = 1e-3
            kwargs_dict["atol"] = 1e-6

            beam_me_up(**kwargs_dict)

            counter = counter + 1
            print(
                "Sweep completion:"
                + str(counter)
                + " of "
                + str(len(equil_times) * len(mirror_rotations) * len(launch_freqs_GHz))
            )
