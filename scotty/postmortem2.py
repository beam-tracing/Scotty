# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 20:15:24 2019

@author: VH Chen
"""

import numpy as np
import matplotlib.pyplot as plt
from scotty.fun_general import (
    find_waist,
    find_distance_from_waist,
    find_q_lab_Cartesian,
    find_nearest,
    contract_special,
)

# import tikzplotlib
import sys


def plot(suffix=""):
    loadfile = np.load("data_output" + suffix + ".npz")
    tau_array = loadfile["tau_array"]
    q_R_array = loadfile["q_R_array"]
    q_zeta_array = loadfile["q_zeta_array"]
    q_Z_array = loadfile["q_Z_array"]
    K_R_array = loadfile["K_R_array"]
    # K_zeta_array = loadfile['K_zeta_array']
    K_Z_array = loadfile["K_Z_array"]
    Psi_3D_output = loadfile["Psi_3D_output"]
    # Psi_w_xx_array = loadfile['Psi_w_xx_array']
    # Psi_w_xy_array = loadfile['Psi_w_xy_array']
    # Psi_w_yy_array = loadfile['Psi_w_yy_array']
    # Psi_3D_output = loadfile['Psi_3D_output']
    # x_hat_Cartesian_output = loadfile['x_hat_Cartesian_output']
    # y_hat_Cartesian_output = loadfile['y_hat_Cartesian_output']
    # b_hat_Cartesian_output = loadfile['b_hat_Cartesian_output']
    x_hat_output = loadfile["x_hat_output"]
    y_hat_output = loadfile["y_hat_output"]
    B_R_output = loadfile["B_R_output"]
    B_T_output = loadfile["B_T_output"]
    B_Z_output = loadfile["B_Z_output"]
    # B_total_output = loadfile['B_total_output']
    # grad_bhat_output = loadfile['grad_bhat_output']
    # g_hat_Cartesian_output = loadfile['g_hat_Cartesian_output']
    g_hat_output = loadfile["g_hat_output"]
    # g_magnitude_output = loadfile['g_magnitude_output']
    # theta_output = loadfile['theta_output']
    # d_theta_d_tau_output = loadfile['d_theta_d_tau_output']
    # d_theta_m_d_tau_array = loadfile['d_theta_m_d_tau_array']
    # kappa_dot_xhat_output = loadfile['kappa_dot_xhat_output']
    # kappa_dot_yhat_output = loadfile['kappa_dot_yhat_output']
    # d_xhat_d_tau_dot_yhat_output = loadfile['d_xhat_d_tau_dot_yhat_output']
    # xhat_dot_grad_bhat_dot_xhat_output = loadfile['xhat_dot_grad_bhat_dot_xhat_output']
    # xhat_dot_grad_bhat_dot_yhat_output = loadfile['xhat_dot_grad_bhat_dot_yhat_output']
    # xhat_dot_grad_bhat_dot_ghat_output = loadfile['xhat_dot_grad_bhat_dot_ghat_output']
    # yhat_dot_grad_bhat_dot_xhat_output = loadfile['yhat_dot_grad_bhat_dot_xhat_output']
    # yhat_dot_grad_bhat_dot_yhat_output = loadfile['yhat_dot_grad_bhat_dot_yhat_output']
    # yhat_dot_grad_bhat_dot_ghat_output = loadfile['yhat_dot_grad_bhat_dot_ghat_output']
    # tau_start = loadfile['tau_start']
    # tau_end = loadfile['tau_end']
    # tau_nu_index=loadfile['tau_nu_index']
    # tau_0_index=loadfile['tau_0_index']
    # K_g_array = loadfile['K_g_array']
    # d_K_g_d_tau_array = loadfile['d_K_g_d_tau_array']
    # B_total_output = loadfile['B_total_output']
    # b_hat_output = loadfile['b_hat_output']
    # gradK_grad_H_output = loadfile['gradK_grad_H_output']
    # gradK_gradK_H_output = loadfile['gradK_gradK_H_output']
    # grad_grad_H_output = loadfile['grad_grad_H_output']
    dH_dR_output = loadfile["dH_dR_output"]
    dH_dZ_output = loadfile["dH_dZ_output"]
    dH_dKR_output = loadfile["dH_dKR_output"]
    dH_dKzeta_output = loadfile["dH_dKzeta_output"]
    dH_dKZ_output = loadfile["dH_dKZ_output"]
    # dB_dR_FFD_debugging = loadfile['dB_dR_FFD_debugging']
    # dB_dZ_FFD_debugging = loadfile['dB_dZ_FFD_debugging']
    # d2B_dR2_FFD_debugging = loadfile['d2B_dR2_FFD_debugging']
    # d2B_dZ2_FFD_debugging = loadfile['d2B_dZ2_FFD_debugging']
    # d2B_dR_dZ_FFD_debugging = loadfile['d2B_dR_dZ_FFD_debugging']
    # poloidal_flux_debugging_1R = loadfile['poloidal_flux_debugging_1R']
    # poloidal_flux_debugging_2R = loadfile['poloidal_flux_debugging_2R']
    # poloidal_flux_debugging_3R = loadfile['poloidal_flux_debugging_2R']
    # poloidal_flux_debugging_1Z = loadfile['poloidal_flux_debugging_1Z']
    # poloidal_flux_debugging_2Z = loadfile['poloidal_flux_debugging_2Z']
    # poloidal_flux_debugging_3Z = loadfile['poloidal_flux_debugging_2Z']
    # poloidal_flux_debugging_2R_2Z = loadfile['poloidal_flux_debugging_2R_2Z']
    # electron_density_debugging_1R = loadfile['electron_density_debugging_1R']
    # electron_density_debugging_2R = loadfile['electron_density_debugging_2R']
    # electron_density_debugging_3R = loadfile['electron_density_debugging_3R']
    # electron_density_debugging_1Z = loadfile['electron_density_debugging_1Z']
    # electron_density_debugging_2Z = loadfile['electron_density_debugging_2Z']
    # electron_density_debugging_3Z = loadfile['electron_density_debugging_3Z']
    # electron_density_debugging_2R_2Z = loadfile['electron_density_debugging_2R_2Z']
    # electron_density_output=loadfile['electron_density_output']
    poloidal_flux_output = loadfile["poloidal_flux_output"]
    dpolflux_dR_debugging = loadfile["dpolflux_dR_debugging"]
    dpolflux_dZ_debugging = loadfile["dpolflux_dZ_debugging"]
    # d2polflux_dR2_FFD_debugging = loadfile['d2polflux_dR2_FFD_debugging']
    # d2polflux_dZ2_FFD_debugging = loadfile['d2polflux_dZ2_FFD_debugging']
    epsilon_para_output = loadfile["epsilon_para_output"]
    epsilon_perp_output = loadfile["epsilon_perp_output"]
    epsilon_g_output = loadfile["epsilon_g_output"]
    loadfile.close()

    loadfile = np.load("data_input" + suffix + ".npz")
    data_poloidal_flux_grid = loadfile["poloidalFlux_grid"]
    data_R_coord = loadfile["data_R_coord"]
    data_Z_coord = loadfile["data_Z_coord"]
    launch_position = loadfile["launch_position"]
    launch_beam_width = loadfile["launch_beam_width"]
    # launch_beam_curvature = loadfile['launch_beam_curvature']
    loadfile.close()

    [q_X_array, q_Y_array, q_Z_array] = find_q_lab_Cartesian(
        np.array([q_R_array, q_zeta_array, q_Z_array])
    )
    numberOfDataPoints = np.size(q_R_array)

    """
    Beam and ray path
    """

    stop_index = numberOfDataPoints

    ## For plotting the plasma in the toroidal plane
    launch_position_X, launch_position_Y, launch_position_Z = find_q_lab_Cartesian(
        launch_position
    )
    entry_position_X, entry_position_Y, entry_position_Z = find_q_lab_Cartesian(
        np.array([q_R_array[0], q_zeta_array[0], q_Z_array[0]])
    )

    plt.figure(figsize=(5, 5))
    plt.title("Poloidal Plane")
    contour_levels = np.linspace(0, 1, 11)
    CS = plt.contour(
        data_R_coord,
        data_Z_coord,
        np.transpose(data_poloidal_flux_grid),
        contour_levels,
        vmin=0,
        vmax=1.2,
        cmap="inferno",
    )
    plt.clabel(
        CS,
        inline=True,
        fontsize=10,
        inline_spacing=-5,
        fmt="%1.1f",
        use_clabeltext=True,
    )  # Labels the flux surfaces
    # plt.xlim(1.0,1.8)
    # plt.ylim(-0.7,0.1)
    plt.plot(q_R_array[:stop_index], q_Z_array[:stop_index], "k")
    plt.plot(
        [launch_position[0], q_R_array[0]], [launch_position[2], q_Z_array[0]], ":k"
    )
    plt.xlim(data_R_coord[0], data_R_coord[-1])
    plt.ylim(data_Z_coord[0], data_Z_coord[-1])

    plt.xlabel("R / m")
    plt.ylabel("Z / m")
    plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.savefig("propagation_poloidal.jpg", dpi=200)

    plt.figure(figsize=(5, 5))
    plt.title("Toroidal Plane")
    plt.xlim(1.0, 1.8)
    plt.ylim(-0.2, 0.6)
    plt.plot(q_X_array[:stop_index], q_Y_array[:stop_index], "k")
    plt.plot(
        [launch_position_X, entry_position_X],
        [launch_position_Y, entry_position_Y],
        ":k",
    )
    # plt.xlim(-data_R_coord[-1],data_R_coord[-1])
    # plt.ylim(-data_R_coord[-1],data_R_coord[-1])
    plt.xlabel("X / m")
    plt.ylabel("Y / m")
    plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)
    plt.gca().set_aspect("equal", adjustable="box")
    # plt.savefig('propagation_toroidal.jpg',dpi=200)

    plt.figure()
    plt.subplot(2, 3, 1)
    plt.plot(tau_array[:stop_index], q_R_array[:stop_index], "r")
    # plt.axvline(tau_array[:stop_index]1[in_index],color='k')
    # plt.axvline(tau_array[:stop_index]1[out_index],color='k')
    plt.title("R")

    plt.subplot(2, 3, 2)
    plt.plot(tau_array[:stop_index], q_zeta_array[:stop_index], "r")
    # plt.axvline(tau_array[:stop_index]1[in_index],color='k')
    # plt.axvline(tau_array[:stop_index]1[out_index],color='k')
    plt.title("zeta")

    plt.subplot(2, 3, 3)
    plt.plot(tau_array[:stop_index], q_Z_array[:stop_index], "r")
    # plt.axvline(tau_array[:stop_index]1[in_index],color='k')
    # plt.axvline(tau_array[:stop_index]1[out_index],color='k')
    plt.title("Z")

    plt.subplot(2, 3, 4)
    plt.plot(tau_array[:stop_index], K_R_array[:stop_index], "r")
    # plt.axvline(tau_array[:stop_index]1[in_index],color='k')
    # plt.axvline(tau_array[:stop_index]1[out_index],color='k')
    plt.title(r"$K_R$")

    plt.subplot(2, 3, 5)

    plt.subplot(2, 3, 6)
    plt.plot(tau_array[:stop_index], K_Z_array[:stop_index], "r")
    # plt.axvline(tau_array[:stop_index]1[in_index],color='k')
    # plt.axvline(tau_array[:stop_index]1[out_index],color='k')
    plt.title(r"$K_Z$")

    plt.figure()
    plt.subplot(3, 3, 1)
    plt.plot(tau_array[:stop_index], np.imag(Psi_3D_output)[:stop_index, 0, 0], "r")
    plt.subplot(3, 3, 2)
    plt.plot(tau_array[:stop_index], np.imag(Psi_3D_output)[:stop_index, 0, 1], "r")
    plt.subplot(3, 3, 3)
    plt.plot(tau_array[:stop_index], np.imag(Psi_3D_output)[:stop_index, 0, 2], "r")
    plt.subplot(3, 3, 4)
    # plt.plot(tau_array[:stop_index],np.imag(Psi_3D_output)[:,1,0],'r')
    plt.subplot(3, 3, 5)
    plt.plot(tau_array[:stop_index], np.imag(Psi_3D_output)[:stop_index, 1, 1], "r")
    plt.subplot(3, 3, 6)
    plt.plot(tau_array[:stop_index], np.imag(Psi_3D_output)[:stop_index, 1, 2], "r")
    plt.subplot(3, 3, 7)
    # plt.plot(tau_array[:stop_index],np.imag(Psi_3D_output)[:,2,0],'r')
    plt.subplot(3, 3, 8)
    # plt.plot(tau_array[:stop_index],np.imag(Psi_3D_output)[:,2,1],'r')
    plt.subplot(3, 3, 9)
    plt.plot(tau_array[:stop_index], np.imag(Psi_3D_output)[:stop_index, 2, 2], "r")

    plt.figure()
    plt.subplot(3, 3, 1)
    plt.plot(tau_array[:stop_index], np.real(Psi_3D_output)[:stop_index, 0, 0], "r")
    plt.subplot(3, 3, 2)
    plt.plot(tau_array[:stop_index], np.real(Psi_3D_output)[:stop_index, 0, 1], "r")
    plt.subplot(3, 3, 3)
    plt.plot(tau_array[:stop_index], np.real(Psi_3D_output)[:stop_index, 0, 2], "r")
    plt.subplot(3, 3, 4)
    # plt.plot(tau_array[:stop_index],np.imag(Psi_3D_output)[:,1,0],'r')
    plt.subplot(3, 3, 5)
    plt.plot(tau_array[:stop_index], np.real(Psi_3D_output)[:stop_index, 1, 1], "r")
    plt.subplot(3, 3, 6)
    plt.plot(tau_array[:stop_index], np.real(Psi_3D_output)[:stop_index, 1, 2], "r")
    plt.subplot(3, 3, 7)
    # plt.plot(tau_array[:stop_index],np.imag(Psi_3D_output)[:,2,0],'r')
    plt.subplot(3, 3, 8)
    # plt.plot(tau_array[:stop_index],np.imag(Psi_3D_output)[:,2,1],'r')
    plt.subplot(3, 3, 9)
    plt.plot(tau_array[:stop_index], np.real(Psi_3D_output)[:stop_index, 2, 2], "r")

    plt.figure()
    plt.subplot(2, 3, 1)
    plt.plot(tau_array[:stop_index], dH_dR_output[:stop_index], "ro")
    # plt.axvline(tau_array[:stop_index]1[in_index],color='k')
    # plt.axvline(tau_array[:stop_index]1[out_index],color='k')
    plt.title(r"$dH_dR$")

    plt.subplot(2, 3, 2)

    plt.subplot(2, 3, 3)
    plt.plot(tau_array[:stop_index], dH_dZ_output[:stop_index], "r")
    # plt.axvline(tau_array[:stop_index]1[in_index],color='k')
    # plt.axvline(tau_array[:stop_index]1[out_index],color='k')
    plt.title(r"$dH_dZ$")

    plt.subplot(2, 3, 4)
    plt.plot(tau_array[:stop_index], dH_dKR_output[:stop_index], "r")
    # plt.axvline(tau_array[:stop_index]1[in_index],color='k')
    # plt.axvline(tau_array[:stop_index]1[out_index],color='k')
    plt.title(r"$dH_dKR$")

    plt.subplot(2, 3, 5)
    plt.plot(tau_array[:stop_index], dH_dKzeta_output[:stop_index], "r")
    # plt.axvline(tau_array[:stop_index]1[in_index],color='k')
    # plt.axvline(tau_array[:stop_index]1[out_index],color='k')
    plt.title(r"$dH_dKzeta$")

    plt.subplot(2, 3, 6)
    plt.plot(tau_array[:stop_index], dH_dKZ_output[:stop_index], "r")
    # plt.axvline(tau_array[:stop_index]1[in_index],color='k')
    # plt.axvline(tau_array[:stop_index]1[out_index],color='k')
    plt.title(r"$dH_dKZ$")

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(tau_array[:stop_index], dpolflux_dR_debugging)

    plt.subplot(1, 2, 2)
    plt.plot(tau_array[:stop_index], dpolflux_dZ_debugging)

    sys.exit()

    plt.figure()
    plt.subplot(3, 3, 1)
    plt.plot(tau_array[:stop_index], gradK_grad_H_output[:stop_index, 0, 0], "r")
    plt.subplot(3, 3, 2)
    plt.subplot(3, 3, 3)
    plt.plot(tau_array[:stop_index], gradK_grad_H_output[:stop_index, 0, 2], "r")
    plt.subplot(3, 3, 4)
    plt.plot(tau_array[:stop_index], gradK_grad_H_output[:stop_index, 1, 0], "r")
    plt.subplot(3, 3, 5)
    plt.subplot(3, 3, 6)
    plt.plot(tau_array[:stop_index], gradK_grad_H_output[:stop_index, 1, 2], "r")
    plt.subplot(3, 3, 7)
    plt.plot(tau_array[:stop_index], gradK_grad_H_output[:stop_index, 2, 0], "r")
    plt.subplot(3, 3, 8)
    plt.subplot(3, 3, 9)
    plt.plot(tau_array[:stop_index], gradK_grad_H_output[:stop_index, 2, 2], "r")

    plt.figure()
    plt.subplot(3, 3, 1)
    plt.plot(tau_array[:stop_index], grad_grad_H_output[:stop_index, 0, 0], "or")
    plt.title("d2H_dR2")
    plt.subplot(3, 3, 2)
    plt.subplot(3, 3, 3)
    plt.plot(tau_array[:stop_index], grad_grad_H_output[:stop_index, 0, 2], "r")
    plt.title("d2H_dR_dZ")
    plt.subplot(3, 3, 4)
    plt.subplot(3, 3, 5)
    plt.subplot(3, 3, 6)
    plt.subplot(3, 3, 7)
    # plt.plot(tau_array[:stop_index],grad_grad_H_output[:stop_index,2,0],'r')
    plt.subplot(3, 3, 8)
    plt.subplot(3, 3, 9)
    plt.plot(tau_array[:stop_index], grad_grad_H_output[:stop_index, 2, 2], "r")
    plt.title("d2H_dZ2")

    plt.figure()
    plt.subplot(3, 3, 1)
    plt.plot(tau_array[:stop_index], gradK_gradK_H_output[:stop_index, 0, 0], "r")
    plt.subplot(3, 3, 2)
    plt.plot(tau_array[:stop_index], gradK_gradK_H_output[:stop_index, 0, 1], "r")
    plt.subplot(3, 3, 3)
    plt.plot(tau_array[:stop_index], gradK_gradK_H_output[:stop_index, 0, 2], "r")
    plt.subplot(3, 3, 4)
    plt.subplot(3, 3, 5)
    plt.plot(tau_array[:stop_index], gradK_gradK_H_output[:stop_index, 1, 1], "r")
    plt.subplot(3, 3, 6)
    plt.plot(tau_array[:stop_index], gradK_gradK_H_output[:stop_index, 1, 2], "r")
    plt.subplot(3, 3, 7)
    plt.subplot(3, 3, 8)
    plt.subplot(3, 3, 9)
    plt.title("d2H_dKZ2")
    plt.plot(tau_array[:stop_index], gradK_gradK_H_output[:stop_index, 2, 2], "r")

    # plt.figure(figsize=(15,5))
    # plt.subplot(2,3,1)
    # plt.plot(tau_array[:stop_index],np.real(Psi_xx_output),'k')
    # plt.plot(tau_array[:stop_index],np.real(M_xx_output),'r')
    # plt.subplot(2,3,2)
    # plt.plot(tau_array[:stop_index],np.real(Psi_xy_output),'k')
    # plt.plot(tau_array[:stop_index],np.real(M_xy_output),'r')
    # plt.subplot(2,3,3)
    # plt.plot(tau_array[:stop_index],np.real(Psi_yy_output),'k')
    # plt.subplot(2,3,4)
    # plt.plot(tau_array[:stop_index],np.imag(Psi_xx_output),'k')
    # plt.plot(tau_array[:stop_index],np.imag(M_xx_output),'r')
    # plt.subplot(2,3,5)
    # plt.plot(tau_array[:stop_index],np.imag(Psi_xy_output),'k')
    # plt.plot(tau_array[:stop_index],np.imag(M_xy_output),'r')
    # plt.subplot(2,3,6)
    # plt.plot(tau_array[:stop_index],np.imag(Psi_yy_output),'k')

    plt.figure(figsize=(15, 5))
    plt.subplot(2, 3, 1)
    plt.plot(tau_array[:stop_index], dB_dR_FFD_debugging[:stop_index], "k")
    plt.title("dB_dR")
    plt.subplot(2, 3, 2)
    plt.plot(tau_array[:stop_index], dB_dZ_FFD_debugging[:stop_index], "k")
    plt.title("dB_dZ")
    plt.subplot(2, 3, 3)
    plt.subplot(2, 3, 4)
    plt.plot(tau_array[:stop_index], d2B_dR2_FFD_debugging[:stop_index], "k")
    plt.title("d2B_dR2")
    plt.subplot(2, 3, 5)
    plt.plot(tau_array[:stop_index], d2B_dZ2_FFD_debugging[:stop_index], "k")
    plt.title("d2B_dZ2")
    plt.subplot(2, 3, 6)
    plt.plot(tau_array[:stop_index], d2B_dR_dZ_FFD_debugging[:stop_index], "k")
    plt.title("d2B_dR_dZ")

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(tau_array[:stop_index], dpolflux_dR_FFD_debugging[:stop_index], "k")
    plt.subplot(2, 2, 2)
    plt.plot(tau_array[:stop_index], dpolflux_dZ_FFD_debugging[:stop_index], "k")
    plt.subplot(2, 2, 3)
    plt.plot(tau_array[:stop_index], d2polflux_dR2_FFD_debugging[:stop_index], "k")
    plt.subplot(2, 2, 4)
    plt.plot(tau_array[:stop_index], d2polflux_dZ2_FFD_debugging[:stop_index], "k")


if __name__ == "__main__":
    plot()
