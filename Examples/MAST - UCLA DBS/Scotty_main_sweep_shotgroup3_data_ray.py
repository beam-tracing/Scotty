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


equil_times = np.array([0.167, 0.179, 0.192, 0.200, 0.217])
mirror_rotations = np.linspace(0, 6, 61)
mirror_tilts = np.array([4.0])
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
    ]
)
# launch_freqs_GHz = np.array([30.0,32.5,35.0,37.5,42.5,45.0,47.5,50.0,55.0,57.5,60.0,62.5])
# shot_array2A = np.array([29677, 29678, 29679, 29681, 29683, 29684])
# shot_array2B = np.array([29692, 29693])

q_R_all = np.zeros(
    [len(equil_times), len(mirror_tilts), len(mirror_rotations), len(launch_freqs_GHz)]
)
q_Z_all = np.zeros_like(q_R_all)
K_mag_all = np.zeros_like(q_R_all)
polflux_all = np.zeros_like(q_R_all)
theta_m_all = np.zeros_like(q_R_all)

total_simulations = (
    len(equil_times) * len(mirror_tilts) * len(mirror_rotations) * len(launch_freqs_GHz)
)
counter = 0
for ii, equil_time in enumerate(equil_times):
    for jj, mirror_tilt in enumerate(mirror_tilts):
        for kk, mirror_rotation in enumerate(mirror_rotations):
            for ll, launch_freq_GHz in enumerate(launch_freqs_GHz):

                kwargs_dict = get_parameters_for_Scotty(
                    "DBS_NSTX_MAST",
                    launch_freq_GHz=launch_freq_GHz,
                    mirror_rotation=mirror_rotation,  # angle, in deg
                    mirror_tilt=mirror_tilt,  # angle, in deg
                    find_B_method="EFITpp",  # EFITpp, UDA_saved, UDA, torbeam
                    equil_time=equil_time,
                    shot=29684,
                    user="Valerian_laptop",
                )
                if kwargs_dict["launch_freq_GHz"] > 52.5:
                    kwargs_dict["mode_flag"] = -1
                else:
                    kwargs_dict["mode_flag"] = 1

                if kwargs_dict["mode_flag"] == 1:
                    mode_string = "O"
                elif kwargs_dict["mode_flag"] == -1:
                    mode_string = "X"

                kwargs_dict["output_filename_suffix"] = (
                    "_t"
                    + f"{mirror_tilt:.1f}"
                    + "_r"
                    + f"{mirror_rotation:.1f}"
                    + "_f"
                    + f"{launch_freq_GHz:.1f}"
                    + "_"
                    + mode_string
                    + "_"
                    + f"{equil_time*1000:.3g}"
                    + "ms"
                )

                kwargs_dict["quick_run"] = True
                kwargs_dict["figure_flag"] = False
                kwargs_dict["Psi_BC_flag"] = False
                kwargs_dict[
                    "output_path"
                ] = "C:\\Dropbox\\VHChen2021\\Data - Scotty\\Run 23\\"
                kwargs_dict["density_fit_parameters"] = None
                kwargs_dict[
                    "ne_data_path"
                ] = "C:\\Dropbox\\VHChen2021\\Data - Equilibrium\MAST\\"
                # kwargs_dict['magnetic_data_path'] = 'C:\\Dropbox\\VHChen2021\\Data - Equilibrium\MAST-U\\'
                kwargs_dict["input_filename_suffix"] = (
                    "_shotgroup3_avr_" + f"{equil_time*1000:.0f}" + "ms"
                )

                if equil_time == 0.167:
                    kwargs_dict["poloidal_flux_enter"] = 1.14071538**2
                elif equil_time == 0.179:
                    kwargs_dict["poloidal_flux_enter"] = 1.14703219**2
                elif equil_time == 0.192:
                    kwargs_dict["poloidal_flux_enter"] = 1.13483721**2
                elif equil_time == 0.200:
                    kwargs_dict["poloidal_flux_enter"] = 1.14390669**2
                elif equil_time == 0.217:
                    kwargs_dict["poloidal_flux_enter"] = 1.14843969**2

                kwargs_dict["delta_R"] = -0.00001
                kwargs_dict["delta_Z"] = -0.00001
                kwargs_dict["delta_K_R"] = 0.01
                kwargs_dict["delta_K_zeta"] = 0.01
                kwargs_dict["delta_K_Z"] = 0.01
                kwargs_dict["interp_smoothing"] = 0.0
                kwargs_dict["len_tau"] = 1002
                kwargs_dict["rtol"] = 1e-4
                kwargs_dict["atol"] = 1e-7

                if ii == 0 and jj == 0 and kk == 0 and kk == ll:
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

                counter = counter + 1
                print("simulation ", counter, "of", total_simulations)

                [q_R, q_Z, K_mag, polflux, theta_m] = beam_me_up(
                    **kwargs_dict
                )

                q_R_all[ii, jj, kk, ll] = q_R
                q_Z_all[ii, jj, kk, ll] = q_Z
                K_mag_all[ii, jj, kk, ll] = K_mag
                polflux_all[ii, jj, kk, ll] = polflux
                theta_m_all[ii, jj, kk, ll] = theta_m

np.savez(
    kwargs_dict["output_path"] + "Cutoff_sweep",
    equil_times=equil_times,
    mirror_tilts=mirror_tilts,
    mirror_rotations=mirror_rotations,
    launch_freqs_GHz=launch_freqs_GHz,
    q_R_all=q_R_all,
    q_Z_all=q_Z_all,
    K_mag_all=K_mag_all,
    polflux_all=polflux_all,
    theta_m_all=theta_m_all,
)
