# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 10:44:34 2018

@author: VH Hall-Chen
Valerian Hongjie Hall-Chen
valerian@hall-chen.com

"""
from scotty.beam_me_up import beam_me_up
import numpy as np

## high density
kwargs_dict = dict(
    [
        ("poloidal_launch_angle_Torbeam", -35.0),
        ("toroidal_launch_angle_Torbeam", -14.5),  # -12.8
        ("launch_freq_GHz", 90.0),
        ("mode_flag", -1),
        ("launch_beam_width", 0.06323503329291348),
        ("launch_beam_curvature", -0.5535179506038995),
        ("launch_position", np.array([4.5, 0.0, -0.6])),
        ##
        ("delta_R", -0.00001),
        ("delta_Z", 0.00001),
        (
            "find_B_method",
            "UDA_saved",
        ),  # either UDA_saved or test_notime for JT60-SA equilibria
        ("Psi_BC_flag", True),
        ("figure_flag", True),
        ("vacuum_propagation_flag", True),
        ("vacuumLaunch_flag", True),
        (
            "density_fit_parameters",
            np.array(
                [
                    1.08424828e+01, 
                    7.78049990e-01,  
                    5.62555233e+00, 
                    5.80966239e+00,
                    -0.1,  
                    2.73776308e-02,  
                    1.15399313e-01,  
                    9.49642550e-01
                ]
            ),
        ),
        ("poloidal_flux_enter", 1.05891249),
        (
            "magnetic_data_path",
            "D:\\Dropbox\\VHChen2021\\Collaborators\\Daniel Carralero\\Processed data\\JT60SA_highden.npz",
        ),
    ]
)

## highbeta
# kwargs_dict = dict(
#     [
#         ("poloidal_launch_angle_Torbeam", -35.0),
#         ("toroidal_launch_angle_Torbeam", -14.5),  # -12.8
#         ("launch_freq_GHz", 90.0),
#         ("mode_flag", -1),
#         ("launch_beam_width", 0.06323503329291348),
#         ("launch_beam_curvature", -0.5535179506038995),
#         ("launch_position", np.array([4.5, 0.0, -0.6])),
#         ##
#         ("delta_R", -0.00001),
#         ("delta_Z", 0.00001),
#         (
#             "find_B_method",
#             "UDA_saved",
#         ),  # either UDA_saved or test_notime for JT60-SA equilibria
#         ("Psi_BC_flag", True),
#         ("figure_flag", True),
#         ("vacuum_propagation_flag", True),
#         ("vacuumLaunch_flag", True),
#         (
#             "ne_data_path",
#             "D:\\Dropbox\\VHChen2022\\Team\\Lim Jiun Yeu\\JT60-SA\\Processed data\\"
#         ),
#         (
#             "input_filename_suffix",
#             "_highbeta"
#         ),
#         ("poloidal_flux_enter", 1.04880885**2),
#         (
#             "magnetic_data_path",
#             "D:\\Dropbox\\VHChen2022\\Team\\Lim Jiun Yeu\\JT60-SA\\Processed data\\JT60SA_highden.npz",
#         ),
#     ]
# )


beam_me_up(**kwargs_dict)
