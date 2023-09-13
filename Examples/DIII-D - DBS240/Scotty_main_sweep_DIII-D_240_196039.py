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

pol_launch_angles = -np.array([6.50,6.33,6.25,6.17,6.08,
                               6.00,5.92,5.83,5.75,5.67,5.50])
tor_launch_angles = -np.array([1.36,0.36,-0.14,-0.64,-1.14,
                               -1.64,-2.14,-2.64,-3.14,-3.64,-4.64])
launch_freqs_GHz = np.array([55.0,62.5,75.0])


equil_time = 1000.0


counter = 0
total_simulations = (
    len(pol_launch_angles) * len(launch_freqs_GHz)
)
for ii, launch_freq_GHz in enumerate(launch_freqs_GHz):
    for jj, pol_launch_angle in enumerate(pol_launch_angles):
        # for kk, tor_launch_angle in enumerate(tor_launch_angles):
        tor_launch_angle = tor_launch_angles[jj]
            
        kwargs_dict = get_parameters_for_Scotty(
            "DBS_UCLA_DIII-D_240",
            launch_freq_GHz=launch_freq_GHz,
            find_B_method="omfit",  # EFITpp, UDA_saved, UDA, torbeam
            equil_time=equil_time,
            shot=196039,
            user="Valerian_desktop",
        )
        
        kwargs_dict["Psi_BC_flag"] = 'discontinuous'
        kwargs_dict["poloidal_launch_angle_Torbeam"] = pol_launch_angle
        kwargs_dict["toroidal_launch_angle_Torbeam"] = tor_launch_angle
        kwargs_dict["mode_flag"] = -1

        kwargs_dict["poloidal_flux_enter"] = 1.0
        kwargs_dict["poloidal_flux_zero_density"] = 1.001
        kwargs_dict["input_filename_suffix"] = "_196039_1000ms"
        kwargs_dict['ne_data_path'] = 'D:\\Dropbox\\VHChen2022\\Data - Equilibrium\\DIII-D\\'
        kwargs_dict['magnetic_data_path'] = 'D:\\Dropbox\\VHChen2022\\Data - Equilibrium\\DIII-D\\'
        kwargs_dict[
            "output_path"
        ] = "D:\\Dropbox\\VHChen2022\\Data - Scotty\\Run 12\\"

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
            "_pol" + f"{pol_launch_angle:.1f}"
            "_tor"
            + f"{tor_launch_angle:.1f}"
            + "_f"
            + f"{kwargs_dict['launch_freq_GHz']:.1f}"
            + "_"
            + mode_string
        )
        kwargs_dict["figure_flag"] = False
        kwargs_dict["detailed_analysis_flag"] = False

        kwargs_dict["verbose_output_flag"] = True

        counter = counter + 1
        print("simulation ", counter, "of", total_simulations)

        an_output = (
            kwargs_dict["output_path"]
            + "data_output"
            + kwargs_dict["output_filename_suffix"]
            + ".npz"
        )
        # if os.path.exists(an_output):
        #     continue
        # else:
        #     beam_me_up(**kwargs_dict)
        beam_me_up(**kwargs_dict)
