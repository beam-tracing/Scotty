# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 10:44:34 2018

@author: VH Hall-Chen
Valerian Hongjie Hall-Chen
valerian@hall-chen.com valerian_hall-chen@ihpc.a-star.edu.sg
"""
from scotty.beam_me_up import beam_me_up
import ast

"""
kwargs_dict = {'poloidal_launch_angle_Torbeam': -2.5,
 'toroidal_launch_angle_Torbeam': -4.7,
 'launch_freq_GHz': 50.0,
 'mode_flag': -1,
 'launch_beam_width': 0.049618973392829176,
 'launch_beam_radius_of_curvature': -0.8075304707139381,
 'launch_position': np.array([2.278, 0.   , 0.   ])}
 """

# loadfile = np.load(fileLocation+'inScotty.npz', allow_pickle=True)
# kwargs_dict = dict(enumerate(loadfile['kwargs_dict'].flatten(),1))[1]
# kwargs_dict = dict(enumerate(loadfile['kwargs_dict'].flatten(),1))[1]
# loadfile.close()


args_data_path = "./"

args_file_name = args_data_path + "argsdict.json"
f = open(args_file_name, "r")
data = f.read()
kwargs_dict = ast.literal_eval(data)
kwargs_dict["figure_flag"] = eval(kwargs_dict["figure_flag"])
kwargs_dict["Psi_BC_flag"] = eval(kwargs_dict["Psi_BC_flag"])
kwargs_dict["vacuum_propagation_flag"] = eval(kwargs_dict["vacuum_propagation_flag"])
kwargs_dict["vacuumLaunch_flag"] = eval(kwargs_dict["vacuumLaunch_flag"])
f.close()

# kwargs_dict = get_parameters_for_Scotty('DBS_synthetic')
# kwargs_dict['output_path'] = './'
beam_me_up(**kwargs_dict)
