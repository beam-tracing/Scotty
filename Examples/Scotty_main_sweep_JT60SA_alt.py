# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 10:44:34 2018

@author: VH Hall-Chen
Valerian Hongjie Hall-Chen
valerian@hall-chen.com

"""
from scotty.beam_me_up import beam_me_up
import numpy as np

pol_launch_angles = np.linspace(-20.0, -40.0, 21)
tor_launch_angles = np.linspace(0, -20, 41)
freqs_GHz = np.linspace(50, 110, 13)

kwargs_dict = dict(
    [
        ("mode_flag", 1),
        ("launch_beam_width", 0.06323503329291348),
        ("launch_beam_curvature", -0.5535179506038995),
        ("launch_position", np.array([4.5, 0.0, -0.6])),
    ]
)

kwargs_dict = dict(
    [
        ("find_B_method", "UDA_saved"),
        ("Psi_BC_flag", True),
        ("figure_flag", False),
        ("vacuum_propagation_flag", True),
        ("vacuumLaunch_flag", True),
        (
            "density_fit_parameters",
            np.array(
                [
                    10.84222049,
                    0.17888095,
                    1.29493525,
                    5.81062798,
                    0.0,
                    0.02727005,
                    0.1152848,
                    0.94942186,
                ]
            ),
        ),
        ("poloidal_flux_enter", 1.3),
        (
            "magnetic_data_path",
            "D:\\Dropbox\\VHChen2021\\Collaborator - Daniel Carralero\\Processed data\\JT60SA_highden.npz",
        ),
    ]
)
kwargs_dict["delta_R"] = -0.00001
kwargs_dict["delta_Z"] = -0.00001
kwargs_dict["delta_K_R"] = 0.1
kwargs_dict["delta_K_zeta"] = 0.1
kwargs_dict["delta_K_Z"] = 0.1
kwargs_dict["interp_smoothing"] = 2.0
kwargs_dict["len_tau"] = 1002
kwargs_dict["rtol"] = 1e-3
kwargs_dict["atol"] = 1e-6
kwargs_dict["detailed_analysis_flag"] = False

counter = 0
for freq_GHz in freqs_GHz:
    for pol_launch_angle in pol_launch_angles:
        for tor_launch_angle in tor_launch_angles:
            kwargs_dict["poloidal_launch_angle_Torbeam"] = pol_launch_angle
            kwargs_dict["toroidal_launch_angle_Torbeam"] = tor_launch_angle
            kwargs_dict["launch_freq_GHz"] = freq_GHz

            if kwargs_dict["mode_flag"] == 1:
                mode_string = "O"
            elif kwargs_dict["mode_flag"] == -1:
                mode_string = "X"

            kwargs_dict["output_filename_suffix"] = (
                "_pol" + f"{pol_launch_angle:.1f}"
                "_tor"
                + f"{tor_launch_angle:.1f}"
                + "_f"
                + f"{freq_GHz:.1f}"
                + "_"
                + mode_string
                + "_"
                + "highdens"
            )
            kwargs_dict[
                "output_path"
            ] = "D:\\Dropbox\\VHChen2021\\Data - Scotty\\Run 14\\"

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
                + str(len(freqs_GHz) * len(pol_launch_angles) * len(tor_launch_angles))
            )
