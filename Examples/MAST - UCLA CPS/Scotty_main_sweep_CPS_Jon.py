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


equil_times = np.array([0.250])
mirror_rotations = np.array([-3.0, -5.0])
mirror_tilts = np.array([-3.0, -5.0])
launch_freqs_GHz = np.array([57.5, 60.0])
# launch_freqs_GHz = np.array([55.0,57.5,60.0,62.5,67.5,70.0,72.5,75.0])


for ii, equil_time in enumerate(equil_times):
    for jj, mirror_rotation in enumerate(mirror_rotations):
        for kk, mirror_tilt in enumerate(mirror_tilts):
            for ll, launch_freq_GHz in enumerate(launch_freqs_GHz):
                for mm, mode_flag in enumerate([-1, 1]):
                    kwargs_dict = get_parameters_for_Scotty(
                        "DBS_NSTX_MAST",
                        launch_freq_GHz=launch_freq_GHz,
                        mirror_rotation=mirror_rotation,  # angle, in deg
                        mirror_tilt=mirror_tilt,  # angle, in deg
                        find_B_method="UDA_saved",  # EFITpp, UDA_saved, UDA, torbeam
                        equil_time=equil_time,
                        shot=30422,
                        user="Valerian_laptop",
                    )

                    kwargs_dict["mode_flag"] = mode_flag

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
                    ] = "C:\\Dropbox\\VHChen2021\\Data - Scotty\\Run 27\\"
                    kwargs_dict["density_fit_parameters"] = None
                    kwargs_dict[
                        "ne_data_path"
                    ] = "C:\\Dropbox\\VHChen2021\\Data - Equilibrium\MAST\\"
                    kwargs_dict["input_filename_suffix"] = (
                        "_CPS_Jon_avr_" + f"{equil_time*1000:.0f}" + "ms"
                    )
                    kwargs_dict[
                        "magnetic_data_path"
                    ] = "C:\\Dropbox\\VHChen2021\\Data - Equilibrium\MAST\\"

                    kwargs_dict["poloidal_flux_enter"] = 1.03857452**2
                    kwargs_dict["Psi_BC_flag"] = True
                    kwargs_dict["vacuum_propagation_flag"] = True

                    kwargs_dict["delta_R"] = -0.0001
                    kwargs_dict["delta_Z"] = -0.0001
                    kwargs_dict["delta_K_R"] = 0.1
                    kwargs_dict["delta_K_zeta"] = 0.1
                    kwargs_dict["delta_K_Z"] = 0.1
                    kwargs_dict["interp_smoothing"] = 0.0
                    kwargs_dict["len_tau"] = 1002
                    kwargs_dict["rtol"] = 1e-3
                    kwargs_dict["atol"] = 1e-6

                    kwargs_dict["verbose_output_flag"] = True
                    # if ii == 0 and jj == 0 and kk == 0 and ll == 0:
                    #     kwargs_dict['verbose_output_flag'] = True
                    # else:
                    #     kwargs_dict['verbose_output_flag'] = False

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

                    # if equil_time == 0.16:
                    #     kwargs_dict['poloidal_flux_enter'] = 1.17280433**2
                    # if equil_time == 0.19:
                    #     kwargs_dict['poloidal_flux_enter'] = 1.20481476**2
                    # if equil_time == 0.22:
                    #     kwargs_dict['poloidal_flux_enter'] = 1.16129185**2
                    # if equil_time == 0.25:
                    #     kwargs_dict['poloidal_flux_enter'] = 1.15082068**2
