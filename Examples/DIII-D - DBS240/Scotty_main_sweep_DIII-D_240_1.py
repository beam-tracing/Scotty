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

pol_launch_angles = np.linspace(-1.6, -15.4, 70)
tor_launch_angles = np.linspace(0, 10.5, 43)
launch_freqs_GHz = np.array([55.0, 57.5, 60.0])

equil_time = 3000.0


for ii, launch_freq_GHz in enumerate(launch_freqs_GHz):
    for jj, pol_launch_angle in enumerate(pol_launch_angles):
        for kk, tor_launch_angle in enumerate(tor_launch_angles):

            kwargs_dict = get_parameters_for_Scotty(
                "DBS_UCLA_DIII-D_240",
                launch_freq_GHz=launch_freq_GHz,
                find_B_method="torbeam",  # EFITpp, UDA_saved, UDA, torbeam
                equil_time=equil_time,
                shot=187103,
                user="Valerian_desktop",
            )

            kwargs_dict["poloidal_launch_angle_Torbeam"] = pol_launch_angle
            kwargs_dict["toroidal_launch_angle_Torbeam"] = tor_launch_angle
            kwargs_dict["mode_flag"] = -1

            kwargs_dict["poloidal_flux_enter"] = 1.44
            kwargs_dict["input_filename_suffix"] = "_187103_3000ms"
            # kwargs_dict['output_path'] = os.path.dirname(os.path.abspath(__file__)) + '\\Output\\'
            kwargs_dict[
                "output_path"
            ] = "D:\\Dropbox\\VHChen2021\\Data - Scotty\\Run 4\\"
            # kwargs_dict['output_path'] = 'C:\\Users\\chenv\\Dropbox\\VHChen2021\\Data - Scotty\\Run 4\\'

            kwargs_dict["delta_R"] = -0.001
            kwargs_dict["delta_Z"] = -0.001
            kwargs_dict["delta_K_R"] = 0.1
            kwargs_dict["delta_K_zeta"] = 0.1
            kwargs_dict["delta_K_Z"] = 0.1

            if kwargs_dict["mode_flag"] == 1:
                mode_string = "O"
            elif kwargs_dict["mode_flag"] == -1:
                mode_string = "X"

            kwargs_dict["output_filename_suffix"] = (
                "_pol" + f"{pol_launch_angle:.2f}"
                "_tor"
                + f"{tor_launch_angle:.2f}"
                + "_f"
                + f"{kwargs_dict['launch_freq_GHz']:.1f}"
                + "_"
                + mode_string
            )
            kwargs_dict["figure_flag"] = False
            kwargs_dict["detailed_analysis_flag"] = False

            if ii == 0 and jj == 0 and kk == 0:
                kwargs_dict["verbose_output_flag"] = True
            else:
                kwargs_dict["verbose_output_flag"] = False

            beam_me_up(**kwargs_dict)
