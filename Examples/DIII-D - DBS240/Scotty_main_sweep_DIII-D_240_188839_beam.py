# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 10:44:34 2018

@author: VH Hall-Chen
Valerian Hongjie Hall-Chen
valerian@hall-chen.com

"""
from scotty.beam_me_up import beam_me_up
import numpy as np
from scotty.init_bruv import get_parameters_for_Scotty

launch_beam_width_scalings = np.linspace(0.90, 1.1, 3)
launch_beam_curvature_scalings = np.array([1.0])
pol_launch_angle = -11.4
tor_launch_angle = 2.2
launch_freqs_GHz = np.array([55.0, 57.5, 60.0, 62.5, 65.0, 67.5, 70.0, 72.5, 75.0])
# launch_freqs_GHz = np.array([55.0,65.0,75.0])


equil_time = 1900.0

counter = 0
for ii, launch_freq_GHz in enumerate(launch_freqs_GHz):
    for jj, launch_beam_width_scaling in enumerate(launch_beam_width_scalings):
        for kk, launch_beam_curvature_scaling in enumerate(
            launch_beam_curvature_scalings
        ):

            kwargs_dict = get_parameters_for_Scotty(
                "DBS_UCLA_DIII-D_240",
                launch_freq_GHz=launch_freq_GHz,
                find_B_method="torbeam",  # EFITpp, UDA_saved, UDA, torbeam
                equil_time=equil_time,
                shot=188839,
                user="Valerian_desktop",
            )

            kwargs_dict["poloidal_launch_angle_Torbeam"] = pol_launch_angle
            kwargs_dict["toroidal_launch_angle_Torbeam"] = tor_launch_angle
            kwargs_dict["mode_flag"] = -1

            kwargs_dict["launch_beam_width"] = (
                launch_beam_width_scaling * kwargs_dict["launch_beam_width"]
            )
            kwargs_dict["launch_beam_curvature"] = (
                launch_beam_curvature_scaling * kwargs_dict["launch_beam_curvature"]
            )

            kwargs_dict["poloidal_flux_enter"] = 1.44
            kwargs_dict["input_filename_suffix"] = "_188839_1900ms"
            # kwargs_dict['output_path'] = os.path.dirname(os.path.abspath(__file__)) + '\\Output\\'
            kwargs_dict[
                "output_path"
            ] = "D:\\Dropbox\\VHChen2021\\Data - Scotty\\Run 17\\"

            kwargs_dict["delta_R"] = -0.00001
            kwargs_dict["delta_Z"] = -0.00001
            kwargs_dict["delta_K_R"] = 0.1
            kwargs_dict["delta_K_zeta"] = 0.1
            kwargs_dict["delta_K_Z"] = 0.1
            kwargs_dict["interp_smoothing"] = 0.0
            kwargs_dict["len_tau"] = 1002
            kwargs_dict["rtol"] = 1e-3
            kwargs_dict["atol"] = 1e-6

            if kwargs_dict["mode_flag"] == 1:
                mode_string = "O"
            elif kwargs_dict["mode_flag"] == -1:
                mode_string = "X"

            kwargs_dict["output_filename_suffix"] = (
                "_w"
                + f"{kwargs_dict['launch_beam_width']:.6f}"
                + "_curv"
                + f"{kwargs_dict['launch_beam_curvature']:.3f}"
                + "_f"
                + f"{kwargs_dict['launch_freq_GHz']:.1f}"
                + "_"
                + mode_string
            )
            kwargs_dict["figure_flag"] = False
            kwargs_dict["detailed_analysis_flag"] = False

            # an_output = kwargs_dict['output_path'] + 'data_output' + kwargs_dict['output_filename_suffix'] + '.npz'
            # if os.path.exists(an_output):
            #     continue
            # else:
            #     beam_me_up(**kwargs_dict)

            if counter == 0:
                kwargs_dict["verbose_output_flag"] = True
            else:
                kwargs_dict["verbose_output_flag"] = False

            beam_me_up(**kwargs_dict)

            counter = counter + 1
            print(
                "Sweep completion:"
                + str(counter)
                + " of "
                + str(
                    len(launch_freqs_GHz)
                    * len(launch_beam_width_scalings)
                    * len(launch_beam_curvature_scalings)
                )
            )

np.savez(
    kwargs_dict["output_path"] + "sweep_summary",
    launch_beam_width_scalings=launch_beam_width_scalings,
    launch_beam_curvature_scalings=launch_beam_curvature_scalings,
    launch_freqs_GHz=launch_freqs_GHz,
    pol_launch_angle=pol_launch_angle,
    tor_launch_angle=tor_launch_angle,
)
