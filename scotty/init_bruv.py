# -*- coding: utf-8 -*-
"""
Created on 26 April 2021

@author: VH Hall-Chen
Valerian Hongjie Hall-Chen
valerian@hall-chen.com

Initialisation.

This file contains the default settings for a range of various cases.
"""
import os
import sys
import math
import numpy as np
from scipy import constants
import pathlib
from netCDF4 import Dataset

from .hornpy import make_my_horn
from .lensalot import make_my_lens
from .fun_general import (
    propagate_beam,
    propagate_circular_beam,
    find_nearest,
    genray_angles_from_mirror_angles,
)


def get_parameters_for_Scotty(
    diagnostic,
    launch_freq_GHz=None,
    mirror_rotation=None,  # angle, in deg
    mirror_tilt=None,  # angle, in deg
    find_B_method=None,
    find_ne_method=None,
    equil_time=None,  # in seconds
    shot=None,
    user=None,
):
    """
    diagnostic:
        DBS_NSTX_MAST
            - Doppler reflectometry (Neal Crocker, Jon Hillesheim, Tony Peebles)
            - Used on MAST, was on loan from NSTX

        DBS_SWIP_MAST-U
            - Doppler reflectometry (Peng Shi)

        DBS_UCLA_MAST-U
            -
        CPS_UCLA_MAST-U
            - This system can either be used in CPS or DBS mode, but not both
              simultaneously (not yet, anyway)

        hiK_Strath_MAST-U
            - High-k scattering diagnostic, Strathclyde (David Speirs, Kevin Ronald)

        DBS_synthetic
            - Circular flux surfaces (not yet implemented)


    field_data_type (equilibrium data):
        torbeam
            - Loads data from ne.dat and topfile. I guess I should implement
              loading for inbeam.dat at some point, too
        UDA
            - Loads EFIT data directly from uda (not yet implemented)
        EFITpp
            - Uses MSE constrained EFIT
        UDA_saved
            - Loads EFIT data from file. UDA data must first be saved to said file
    """
    ## Initialise dictionaries
    kwargs_dict = dict()
    args_dict = {
        "poloidal_launch_angle_Torbeam": None,
        "toroidal_launch_angle_Torbeam": None,
        "launch_freq_GHz": None,
        "mode_flag": None,
        "launch_beam_width": None,
        "launch_beam_curvature": None,
        "launch_position": None,
    }

    # User settings. Currently just loads paths
    ne_path, topfile_path, inbeam_path, efitpp_path, UDA_saved_path = user_settings(
        diagnostic, user, shot
    )

    if find_B_method == "torbeam":
        kwargs_dict["ne_data_path"] = ne_path
        kwargs_dict["magnetic_data_path"] = topfile_path

    ## Assign keys that we already have
    if launch_freq_GHz is not None:
        args_dict["launch_freq_GHz"] = launch_freq_GHz

    if find_B_method is not None:
        kwargs_dict["find_B_method"] = find_B_method

    if equil_time is not None:
        kwargs_dict["equil_time"] = equil_time

    if shot is not None:
        kwargs_dict["shot"] = shot
    ##

    ## Default settings for each diagnostic
    if diagnostic == "DBS_NSTX_MAST":

        args_dict["launch_position"] = np.array(
            [2.43521, 0, 0]
        )  # q_R, q_zeta, q_Z. q_zeta = 0 at launch, by definition
        # args_dict['launch_position'] = np.array([2.278,0,0]) # DBS UCLA MAST-U

        kwargs_dict["Psi_BC_flag"] = True
        kwargs_dict["figure_flag"] = True
        kwargs_dict["vacuum_propagation_flag"] = True
        kwargs_dict["vacuumLaunch_flag"] = True

        # shots_with_efitpp_data = ([
        #                             29904,29905,29906,29908,29910,
        #                             30073,30074,30075,30076,30077,
        #                             30175,30176,30177,30178,30179
        #                            ])
        # if shot in shots_with_efitpp_data:
        #     kwargs_dictionary['find_B_method'] = 'efitpp'
        # else
        #     kwargs_dictionary['find_B_method'] = 'efit'

        ## Density settings
        if (
            (find_B_method == "EFITpp")
            or (find_B_method == "UDA_saved")
            or (find_B_method == "test")
        ):
            ne_fit_param, ne_fit_time, poloidal_flux_enter = ne_settings(
                diagnostic, shot, equil_time, find_ne_method
            )

            kwargs_dict["density_fit_parameters"] = ne_fit_param
            kwargs_dict["poloidal_flux_enter"] = poloidal_flux_enter
            kwargs_dict["ne_data_path"] = UDA_saved_path

        ##

        ## MAST and MAST-U specific B-field and poloidal flux settings
        if find_B_method == "UDA":
            print("Not yet implemented")
            sys.exit()

        elif find_B_method == "EFITpp":

            dataset = Dataset(efitpp_path / "efitOut.nc")
            efitpp_times = dataset.variables["time"][:]
            dataset.close()
            efit_time_index = find_nearest(efitpp_times, equil_time)

            print("Nearest EFIT++ time:", efitpp_times[efit_time_index])

            kwargs_dict["magnetic_data_path"] = efitpp_path

        elif find_B_method == "UDA_saved" and shot <= 30471:  # MAST:
            loadfile = np.load(UDA_saved_path / f"{shot}_equilibrium_data.npz")
            t_base_C = loadfile["t_base_C"]
            loadfile.close()
            efit_time_index = find_nearest(t_base_C, equil_time)

            print("Nearest EFIT time:", t_base_C[efit_time_index])

            kwargs_dict["magnetic_data_path"] = UDA_saved_path

        elif (
            find_B_method == "UDA_saved" and shot > 30471
        ) or find_B_method == "test":  # MAST:
            loadfile = np.load(UDA_saved_path / f"{shot}_equilibrium_data.npz")
            time_EFIT = loadfile["time_EFIT"]
            loadfile.close()
            efit_time_index = find_nearest(time_EFIT, equil_time)

            print("Nearest EFIT time:", time_EFIT[efit_time_index])

            kwargs_dict["magnetic_data_path"] = UDA_saved_path
        ##

        # Convert mirror angles to launch angles
        if (mirror_rotation is not None) and (mirror_tilt is not None):

            (
                toroidal_launch_angle_genray,
                poloidal_launch_angle_genray,
            ) = genray_angles_from_mirror_angles(
                mirror_rotation,
                mirror_tilt,
                offset_for_window_norm_to_R=np.rad2deg(math.atan2(125, 2432)),
            )

            poloidal_launch_angle_Torbeam = -poloidal_launch_angle_genray
            toroidal_launch_angle_Torbeam = -toroidal_launch_angle_genray

            args_dict["poloidal_launch_angle_Torbeam"] = poloidal_launch_angle_Torbeam
            args_dict["toroidal_launch_angle_Torbeam"] = toroidal_launch_angle_Torbeam
        ##

        ## Beam settings
        launch_beam_width, launch_beam_curvature = beam_settings(
            diagnostic, launch_freq_GHz=launch_freq_GHz
        )
        args_dict["launch_beam_width"] = launch_beam_width
        args_dict["launch_beam_curvature"] = launch_beam_curvature
        ##

    elif diagnostic == "DBS_UCLA_MAST-U":
        ## Default settings
        args_dict["launch_position"] = np.array(
            [2.278, 0, 0]
        )  # q_R, q_zeta, q_Z. q_zeta = 0 at launch, by definition
        print("Warning: launch_position is an estimate")
        ## Launch position changes ~1mm based on lens settings
        kwargs_dict["Psi_BC_flag"] = True
        kwargs_dict["figure_flag"] = True
        kwargs_dict["vacuum_propagation_flag"] = True
        kwargs_dict["vacuumLaunch_flag"] = True

        ## Beam settings
        launch_beam_width, launch_beam_curvature = beam_settings(
            diagnostic, launch_freq_GHz=launch_freq_GHz, method="thin_lens"
        )
        args_dict["launch_beam_width"] = launch_beam_width
        args_dict["launch_beam_curvature"] = launch_beam_curvature
        ##

    elif diagnostic == "DBS_SWIP_MAST-U":
        ## Default settings
        args_dict["launch_position"] = np.array(
            [2.43521, 0, 0]
        )  # q_R, q_zeta, q_Z. q_zeta = 0 at launch, by definition
        print("Warning: launch_position is an estimate")
        ## I'm checking what this actually is from Peng. Currently using the
        ## MAST UCLA DBS as a guide
        kwargs_dict["Psi_BC_flag"] = True
        kwargs_dict["figure_flag"] = True
        kwargs_dict["vacuum_propagation_flag"] = True
        kwargs_dict["vacuumLaunch_flag"] = True

        ## Beam settings
        launch_beam_width, launch_beam_curvature = beam_settings(
            diagnostic, launch_freq_GHz=launch_freq_GHz, method="estimate_fix_w0"
        )
        args_dict["launch_beam_width"] = launch_beam_width
        args_dict["launch_beam_curvature"] = launch_beam_curvature

        if (
            (find_B_method == "EFITpp")
            or (find_B_method == "UDA_saved")
            or (find_B_method == "test")
        ):
            ne_fit_param, ne_fit_time, poloidal_flux_enter = ne_settings(
                diagnostic, shot, equil_time, find_ne_method
            )

            kwargs_dict["density_fit_parameters"] = ne_fit_param
            kwargs_dict["poloidal_flux_enter"] = poloidal_flux_enter
            kwargs_dict["ne_data_path"] = UDA_saved_path

        if (
            find_B_method == "UDA_saved" and shot > 30471
        ) or find_B_method == "test":  # MAST:
            loadfile = np.load(UDA_saved_path / f"{shot}_equilibrium_data.npz")
            time_EFIT = loadfile["time_EFIT"]
            loadfile.close()
            efit_time_index = find_nearest(time_EFIT, equil_time)

            print("Nearest EFIT time:", time_EFIT[efit_time_index])

            kwargs_dict["magnetic_data_path"] = UDA_saved_path
        ##

    elif diagnostic == "DBS_UCLA_DIII-D_240":
        ## Default settings
        args_dict["launch_position"] = np.array(
            [2.587, 0, -0.0157]
        )  # q_R, q_zeta, q_Z. q_zeta = 0 at launch, by definition
        kwargs_dict["Psi_BC_flag"] = True
        kwargs_dict["figure_flag"] = True
        kwargs_dict["vacuum_propagation_flag"] = True
        kwargs_dict["vacuumLaunch_flag"] = True

        ## Beam settings
        launch_beam_width, launch_beam_curvature = beam_settings(
            diagnostic, launch_freq_GHz=launch_freq_GHz, method="thin_lens"
        )
        args_dict["launch_beam_width"] = launch_beam_width
        args_dict["launch_beam_curvature"] = launch_beam_curvature
        ##

    elif diagnostic == "DBS_synthetic":
        args_dict["poloidal_launch_angle_Torbeam"] = 6.0
        args_dict["toroidal_launch_angle_Torbeam"] = 0.0
        args_dict["launch_freq_GHz"] = 55.0
        args_dict["mode_flag"] = 1
        args_dict["launch_beam_width"] = 0.04
        args_dict["launch_beam_curvature"] = 1 / -4.0
        args_dict["launch_position"] = np.array(
            [2.587, 0, -0.0157]
        )  # q_R, q_zeta, q_Z. q_zeta = 0 at launch, by definition

        ne_fit_param = np.array([4.0, 1.0])

        kwargs_dict["density_fit_parameters"] = ne_fit_param
        kwargs_dict["find_B_method"] = "analytical"
        kwargs_dict["Psi_BC_flag"] = True
        kwargs_dict["figure_flag"] = False
        kwargs_dict["vacuum_propagation_flag"] = True
        kwargs_dict["vacuumLaunch_flag"] = True
        kwargs_dict["poloidal_flux_enter"] = ne_fit_param[1]
        kwargs_dict["B_T_axis"] = 1.0
        kwargs_dict["B_p_a"] = 0.1
        kwargs_dict["R_axis"] = 1.5
        kwargs_dict["minor_radius_a"] = 0.5

    return args_dict, kwargs_dict


def beam_settings(diagnostic, method="data", launch_freq_GHz=None, beam_data=None):
    """
    method:
        horn_and_lens
            - Uses information about the horn and lens to figure out what
              the launch beam properties should be
        data
            - uses stored data
        experimental_data
            - Uses experimental measurements of the beam properties


    """
    if diagnostic == "DBS_NSTX_MAST":
        if method == "horn_and_lens":
            if launch_freq_GHz > 52.5:
                name = "MAST_V_band"
                horn_to_lens = 0.139  # V Band
                lens_to_mirror = 0.644  # lens to steering mirror

            elif launch_freq_GHz < 52.5:
                name = "MAST_Q_band"
                horn_to_lens = 0.270  # Q Band
                lens_to_mirror = 0.6425  # lens to steering mirror

            myLens = make_my_lens(name)
            myHorn = make_my_horn(name)
            angular_frequency = 2 * np.pi * 10.0**9 * launch_freq_GHz
            wavenumber_K0 = angular_frequency / constants.c

            horn_width, horn_curvature = myHorn.output_beam()

            Psi_w_horn_cartersian = np.array(
                [
                    [wavenumber_K0 * horn_curvature + 2j * horn_width ** (-2), 0],
                    [0, wavenumber_K0 * horn_curvature + 2j * horn_width ** (-2)],
                ]
            )

            Psi_w_lens_cartesian_input = propagate_beam(
                Psi_w_horn_cartersian, horn_to_lens, launch_freq_GHz
            )

            Psi_w_lens_cartesian_output = myLens.output_beam(
                Psi_w_lens_cartesian_input, launch_freq_GHz
            )

            Psi_w_cartesian_launch = propagate_beam(
                Psi_w_lens_cartesian_output, lens_to_mirror, launch_freq_GHz
            )
            launch_beam_width = np.sqrt(2 / np.imag(Psi_w_cartesian_launch[0, 0]))
            launch_beam_curvature = (
                np.real(Psi_w_cartesian_launch[0, 0]) / wavenumber_K0
            )

        if method == "data":
            freqs = np.array(
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
            launch_beam_widths = (
                np.array(
                    [
                        46.90319593,
                        44.8730752,
                        43.03016639,
                        41.40562031,
                        38.50759751,
                        37.65323989,
                        36.80672175,
                        36.29814335,
                        38.43065497,
                        37.00251598,
                        35.72544826,
                        34.57900305,
                        32.61150219,
                        31.76347845,
                        30.99132929,
                        30.28611839,
                    ]
                )
                * 0.001
            )
            launch_beam_radii_of_curvature = (
                np.array(
                    [
                        -9211.13447598,
                        -5327.42027113,
                        -3834.26164617,
                        -2902.09214589,
                        -1961.58420391,
                        -1636.82546574,
                        -1432.59817651,
                        -1296.20353095,
                        -1437.24234181,
                        -1549.7853604,
                        -1683.5681014,
                        -1843.91364265,
                        -2277.59660009,
                        -2577.93648944,
                        -2964.57675092,
                        -3479.14501841,
                    ]
                )
                * 0.001
            )

            freq_idx = find_nearest(freqs, launch_freq_GHz)
            launch_beam_width = launch_beam_widths[freq_idx]
            launch_beam_curvature = 1 / launch_beam_radii_of_curvature[freq_idx]

    elif diagnostic == "DBS_CIEMAT_JT60SA":
        if method == "data":
            # freqs = np.linspace(50,110,13)
            # launch_beam_widths = np.array([
            #     46.90319593, 44.8730752 , 43.03016639, 41.40562031,
            #     38.50759751, 37.65323989, 36.80672175, 36.29814335,
            #     38.43065497, 37.00251598, 35.72544826, 34.57900305,
            #     32.61150219, 31.76347845, 30.99132929, 30.28611839
            #     ])*0.001
            # launch_beam_radii_of_curvature = np.array([
            #     -9211.13447598, -5327.42027113, -3834.26164617, -2902.09214589,
            #     -1961.58420391, -1636.82546574, -1432.59817651, -1296.20353095,
            #     -1437.24234181, -1549.7853604 , -1683.5681014 , -1843.91364265,
            #     -2277.59660009, -2577.93648944, -2964.57675092, -3479.14501841
            #     ])*0.001
            #
            # freq_idx = find_nearest(freqs, launch_freq_GHz)
            # launch_beam_width = launch_beam_widths[freq_idx]
            # launch_beam_curvature = 1/launch_beam_radii_of_curvature[freq_idx]

            # 90 GHz
            launch_beam_width = 0.06323503329291348
            launch_beam_curvature = -0.5535179506038995

    elif diagnostic == "DBS_UCLA_DIII-D_240":
        if method == "thin_lens":
            name = "DBS_UCLA_DIII-D_240"
            ## The lens is directly in front of the waveguide (wg)
            wg_width = 0.025
            wg_curvature = 0.0
            wg_to_mirror = 0.08255

            focal_lengths = np.array(
                [
                    0.45181379,
                    0.46384668,
                    0.47595193,
                    0.488133,
                    0.50039298,
                    0.51273458,
                    0.52516023,
                    0.53767209,
                    0.55027209,
                    0.56296199,
                    0.57574337,
                    0.58861765,
                    0.60158613,
                    0.61465001,
                    0.62781035,
                    0.64106815,
                    0.65442432,
                    0.66787969,
                    0.68143503,
                    0.69509106,
                    0.70884843,
                    0.72270775,
                    0.73666959,
                    0.75073448,
                    0.7649029,
                    0.7791753,
                    0.79355212,
                    0.80803376,
                    0.82262058,
                    0.83731294,
                    0.85211116,
                    0.86701555,
                    0.8820264,
                    0.89714399,
                    0.91236857,
                    0.92770038,
                    0.94313965,
                    0.95868659,
                    0.97434142,
                    0.99010431,
                    1.00597547,
                    1.02195505,
                    1.03804323,
                    1.05424016,
                    1.07054599,
                    1.08696085,
                    1.10348489,
                    1.12011823,
                    1.13686099,
                    1.15371329,
                    1.17067524,
                    1.18774694,
                    1.20492849,
                    1.22222,
                    1.23962155,
                    1.25713322,
                    1.27475512,
                    1.2924873,
                    1.31032986,
                    1.32828286,
                    1.34634637,
                ]
            )
            freqs_GHz = np.linspace(50, 80, 61)
            nearest_freq_idx = find_nearest(freqs_GHz, launch_freq_GHz)
            focal_length = focal_lengths[nearest_freq_idx]

            myLens = make_my_lens(name, lens_type="thin", focal_length=focal_length)
            angular_frequency = 2 * np.pi * 10.0**9 * launch_freq_GHz
            wavenumber_K0 = angular_frequency / constants.c

            Psi_w_wg = np.array(
                [
                    [wavenumber_K0 * wg_curvature + 2j * wg_width ** (-2), 0],
                    [0, wavenumber_K0 * wg_curvature + 2j * wg_width ** (-2)],
                ]
            )

            Psi_w_lens = myLens.output_beam(Psi_w_wg, launch_freq_GHz)

            Psi_w_mirror = propagate_beam(Psi_w_lens, wg_to_mirror, launch_freq_GHz)

            launch_beam_width = np.sqrt(2 / np.imag(Psi_w_mirror[0, 0]))
            launch_beam_curvature = np.real(Psi_w_mirror[0, 0]) / wavenumber_K0

    elif diagnostic == "DBS_UCLA_MAST-U":
        name = "DBS_UCLA_MAST-U"
        myLens = make_my_lens(name)
        angular_frequency = 2 * np.pi * 10.0**9 * launch_freq_GHz
        wavenumber_K0 = angular_frequency / constants.c

        horn_width = 0.0064
        horn_curvature = 0.0
        horn_to_lens = 0.165

        angular_frequency = 2 * np.pi * 10.0**9 * launch_freq_GHz
        wavenumber = angular_frequency / constants.c

        Psi_w_horn_cartersian = np.array(
            [
                [wavenumber * horn_curvature + 2j * horn_width ** (-2), 0],
                [0, wavenumber * horn_curvature + 2j * horn_width ** (-2)],
            ]
        )

        Psi_w_lens_cartesian_input = propagate_beam(
            Psi_w_horn_cartersian, horn_to_lens, launch_freq_GHz
        )

        Psi_w_lens_cartesian_output = myLens.output_beam(
            Psi_w_lens_cartesian_input, launch_freq_GHz
        )
        print(
            "Warning: lens known to change output beam properties depending on its y position, ignoring this effect"
        )
        launch_beam_width = np.sqrt(2 / np.imag(Psi_w_lens_cartesian_output[0, 0]))
        launch_beam_curvature = (
            np.real(Psi_w_lens_cartesian_output[0, 0]) / wavenumber_K0
        )

    elif diagnostic == "DBS_SWIP_MAST-U":
        # if method == 'thin_lens':
        #     name = 'DBS_SWIP_MAST-U0'
        if method == "estimate_var_w0":
            if launch_freq_GHz <= 50.0:  # Q band
                w0 = np.sqrt(launch_freq_GHz / 40) * 0.08
            else:  # V band
                w0 = np.sqrt(launch_freq_GHz / 60) * 0.04
            distance = (
                -0.277
            )  # window to steering mirror, negative because the mirror is behind the window

            launch_angular_frequency = 2 * np.pi * launch_freq_GHz * 10**9
            wavenumber_K0 = launch_angular_frequency / constants.c

            launch_beam_width, launch_beam_curvature = propagate_circular_beam(
                distance, wavenumber_K0, w0
            )

        if method == "estimate_fix_w0":
            if launch_freq_GHz <= 50.0:  # Q band
                w0 = 0.08
            else:  # V band
                w0 = 0.04
            distance = (
                -0.277
            )  # window to steering mirror, negative because the mirror is behind the window

            launch_angular_frequency = 2 * np.pi * launch_freq_GHz * 10**9
            wavenumber_K0 = launch_angular_frequency / constants.c

            launch_beam_width, launch_beam_curvature = propagate_circular_beam(
                distance, wavenumber_K0, w0
            )

    return launch_beam_width, launch_beam_curvature


def ne_settings(diagnostic, shot, time, find_ne_method):
    ne_fit_param = None
    ne_fit_time = None
    poloidal_flux_enter = None
    poloidal_fluxes_enter = None

    if find_ne_method is None or shot is None:
        return ne_fit_param, ne_fit_time, poloidal_flux_enter

    if diagnostic == "DBS_NSTX_MAST":
        if shot == 29684:
            #
            ne_fit_params = np.array(
                [
                    [2.1, -1.9, 1.2],  # 167ms
                    [2.1, -1.9, 1.25],  # 179ms
                    [2.3, -1.9, 1.2],  # 192ms
                    [2.4, -2.0, 1.2],  # 200ms
                    [2.5, -2.1, 1.2],  # 217ms
                ]
            )
            ne_fit_times = np.array([0.167, 0.179, 0.192, 0.200, 0.217])

        elif shot == 29908:
            #
            if find_ne_method == "tanh":
                ne_fit_params = np.array(
                    [
                        [2.3, -1.9, 1.18],  # 150ms
                        [2.55, -2.2, 1.15],  # 160ms
                        [2.8, -2.2, 1.15],  # 170ms
                        [3.0, -2.35, 1.2],  # 180ms
                        [3.25, -2.4, 1.22],  # 190ms
                        [3.7, -2.7, 1.15],  # 200ms
                        [4.2, -2.0, 1.2],  # 210ms
                        [4.5, -1.8, 1.24],  # 220ms
                        [4.8, -1.8, 1.2],  # 230ms
                        [5.2, -1.8, 1.2],  # 240ms
                        [5.2, -2.8, 1.1],  # 250ms
                        [5.7, -1.9, 1.15],  # 260ms
                        [5.8, -2.2, 1.1],  # 270ms
                        [6.5, -1.7, 1.15],  # 280ms
                        [6.6, -1.8, 1.1],  # 290ms
                    ]
                )
                ne_fit_times = np.linspace(0.150, 0.290, 15)

            if find_ne_method == "poly3":
                ne_fit_params = np.array(
                    [
                        [-3.39920666, 3.3767761, -1.55984715, 2.49116064],
                        [-3.31670147, 2.24970438, -0.46971473, 2.47113803],
                        [-3.44610169, 1.69591882, 0.22709583, 2.62872259],
                        [-4.91157473, 3.12397459, 0.30956902, 2.71940548],
                        [-6.99408536, 4.98094795, 0.36188724, 2.86325923],
                        [-4.01026147, 0.89218099, 1.24564799, 3.24225355],
                        [-4.21483706, 1.73180927, 0.63975703, 3.48532344],
                        [-3.89636166, 1.00604874, 0.5745085, 3.85908854],
                        [-4.75241047, 1.34810624, 0.7207821, 4.11300423],
                        [-2.98806121, -1.48680106, 1.51977942, 4.54713015],
                    ]
                )
                poloidal_fluxes_enter = np.array(
                    [
                        1.14908613,
                        1.13336773,
                        1.13850717,
                        1.1274871,
                        1.09851547,
                        1.1302204,
                        1.15828467,
                        1.14394922,
                        1.1153502,
                        1.13408871,
                    ]
                )
                ne_fit_times = np.linspace(0.160, 0.250, 10)

        elif shot == 29980:
            ne_fit_params = np.array([[2.3, -2.6, 1.12]])  # 200ms
            ne_fit_times = np.array([0.200])

        elif shot == 30073 or shot == 30074:  # TODO: check 30074
            # Fit underestimates TS density when polflux < 0.2 (roughly, for some of the times)
            ne_fit_params = np.array(
                [
                    [2.8, -1.4, 1.1],  # 190ms
                    [2.9, -1.4, 1.15],  # 200ms
                    [3.0, -1.3, 1.2],  # 210ms
                    [3.4, -1.2, 1.2],  # 220ms
                    [3.6, -1.2, 1.2],  # 230ms
                    [4.0, -1.2, 1.2],  # 240ms
                    [4.4, -1.2, 1.2],  # 250ms
                ]
            )
            ne_fit_times = np.linspace(0.190, 0.250, 7)

        elif shot == 45091:
            ne_fit_params = np.array(
                [
                    [5.5, -0.6, 1.2],  # 390ms
                    [4.8, -0.6, 1.2],  # 400ms
                    [3.0, -1.3, 1.2],  # 410ms
                ]
            )
            ne_fit_times = np.linspace(0.390, 0.410, 3)

        elif shot == 45154:
            ne_fit_params = np.array(
                [
                    [2.4, -1.8, 1.12],  # 510ms
                ]
            )
            ne_fit_times = np.array([0.510])

        elif shot == 45189:
            ne_fit_params = np.array(
                [[2.3, -1.8, 1.12], [3.5, -1.35, 1.2]]  # 200ms  # 650ms
            )
            ne_fit_times = np.array([0.200, 0.650])

        else:
            print("No fit data saved for shot:", shot)
            sys.exit()

        nearest_time_idx = find_nearest(ne_fit_times, time)

        ne_fit_param = ne_fit_params[nearest_time_idx, :]
        ne_fit_time = ne_fit_times[nearest_time_idx]
        print("Nearest ne fit time:", ne_fit_time)

        if poloidal_fluxes_enter is not None:
            poloidal_flux_enter = poloidal_fluxes_enter[nearest_time_idx]
        else:
            poloidal_flux_enter = ne_fit_params[2]

    elif diagnostic == "DBS_UCLA_DIII-D_240":
        print("Not yet implemented")

    return ne_fit_param, ne_fit_time, poloidal_flux_enter


def user_settings(diagnostic, user, shot):
    """
    Choosing paths appropriately
    """

    # Default path: all input files in current working directory
    default_input_files_path = pathlib.Path(".")

    #########################
    ## Initialising default paths
    ## Paths are overwritten if specific users are chosen
    ne_path = default_input_files_path
    topfile_path = default_input_files_path
    inbeam_path = default_input_files_path
    efitpp_path = default_input_files_path
    UDA_saved_path = default_input_files_path
    #########################

    if user == "Freia":
        ## Not yet properly implemented
        efitpp_path = None

    elif user == "Valerian_desktop" or user == "Valerian_laptop":

        if user == "Valerian_desktop":
            prefix = pathlib.Path("D:\\Dropbox\\")
        elif user == "Valerian_laptop":
            prefix = pathlib.Path("C:\\Dropbox\\")

        if diagnostic == "DBS_NSTX_MAST" or diagnostic == "DBS_SWIP_MAST-U":
            if shot == 29684:
                # MAST reruns of EFIT. Done by Lucy Kogan.
                # 29684: no MSE data, but reprocessed with more constraints, only good at the edge
                efitpp_path = (
                    prefix
                    / f"VHChen2020/Data/Equilibrium/MAST/Lucy_EFIT_runs/{shot}/epk_lkogan_01/"
                )
                ne_path = prefix / "VHChen2021/Data - Equilibrium/MAST/"

            elif shot in [30073, 30074, 30075, 30076, 30077]:
                # MAST reruns of EFIT. Done by Lucy Kogan.
                # 30073--30077: MSE data, processed better than original runs
                efitpp_path = (
                    prefix
                    / f"VHChen2020/Data/Equilibrium/MAST/Lucy_EFIT_runs/{shot}/epi_lkogan_01/"
                )
            elif shot in [29908]:
                # MAST EFIT runs. List of available shots not updated.
                efitpp_path = (
                    prefix
                    / f"VHChen2020/Data/Equilibrium/MAST/MSE_efitruns/{shot}/Pass0/"
                )
            ## If it's not any of the above shots, I'll assume that there's no efit++ data
            elif shot > 30471:  # MAST-U
                UDA_saved_path = (
                    prefix / "VHChen2020/Data/Equilibrium/MAST-U/Equilibrium_pyuda/"
                )
            else:
                UDA_saved_path = (
                    prefix / "VHChen2020/Data/Equilibrium/MAST/Equilibrium_pyuda/"
                )

        elif diagnostic == "DBS_UCLA_DIII-D_240":
            ne_path = prefix / "VHChen2021/Data - Equilibrium/DIII-D/"
            topfile_path = prefix / "VHChen2021/Data - Equilibrium/DIII-D/"

    return ne_path, topfile_path, inbeam_path, efitpp_path, UDA_saved_path
