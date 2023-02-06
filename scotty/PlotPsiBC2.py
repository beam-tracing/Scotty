# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 12:06:45 2020

@author: VH Chen
"""


import numpy as np
import matplotlib.pyplot as plt

from scotty.fun_general import (
    find_widths_and_curvatures,
    find_Psi_3D_lab_Cartesian,
    find_Psi_3D_lab,
)


def plot():
    loadfile = np.load("data_input0.npz")
    launch_position = loadfile["launch_position"]
    launch_K = loadfile["launch_K"]
    loadfile.close()

    loadfile = np.load("data_output0.npz")
    g_magnitude_output = loadfile["g_magnitude_output"]
    q_R_array = loadfile["q_R_array"]
    q_zeta_array = loadfile["q_zeta_array"]
    K_R_array = loadfile["K_R_array"]
    K_zeta_initial = loadfile["K_zeta_initial"]
    K_Z_array = loadfile["K_Z_array"]
    b_hat_output = loadfile["b_hat_output"]
    g_hat_output = loadfile["g_hat_output"]
    x_hat_output = loadfile["x_hat_output"]
    y_hat_output = loadfile["y_hat_output"]
    Psi_3D_output = loadfile["Psi_3D_output"]
    Psi_3D_lab_launch = loadfile["Psi_3D_lab_launch"]
    dH_dKR_output = loadfile["dH_dKR_output"]
    dH_dKzeta_output = loadfile["dH_dKzeta_output"]
    dH_dKZ_output = loadfile["dH_dKZ_output"]
    loadfile.close()

    loadfile = np.load("analysis_output0.npz")
    Psi_xx_output = loadfile["Psi_xx_output"]
    Psi_xy_output = loadfile["Psi_xy_output"]
    Psi_yy_output = loadfile["Psi_yy_output"]
    Psi_xx_entry = loadfile["Psi_xx_entry"]
    Psi_xy_entry = loadfile["Psi_xy_entry"]
    Psi_yy_entry = loadfile["Psi_yy_entry"]
    K_magnitude_array = loadfile["K_magnitude_array"]
    loadfile.close()

    K_magnitude_entry = K_magnitude_array[0]

    (
        widths_entry,
        widths_entry_eigvec,
        curvatures_entry,
        curvatures_entry_eigvec,
    ) = find_widths_and_curvatures(
        Psi_xx_entry, Psi_xy_entry, Psi_yy_entry, K_magnitude_entry
    )
    (
        widths_initial,
        widths_initial_eigvec,
        curvatures_initial,
        curvatures_initial_eigvec,
    ) = find_widths_and_curvatures(
        Psi_xx_output[0], Psi_xy_output[0], Psi_yy_output[0], K_magnitude_array[0]
    )

    W_xx_entry = np.sqrt(2 / np.imag(Psi_xx_entry))
    W_xy_entry = np.sign(np.imag(Psi_xy_entry)) * np.sqrt(
        2 / abs(np.imag(Psi_xy_entry))
    )
    W_yy_entry = np.sqrt(2 / np.imag(Psi_yy_entry))
    R_xx_entry = K_magnitude_entry / np.real(Psi_xx_entry)
    R_xy_entry = K_magnitude_entry / np.real(Psi_xy_entry)
    R_yy_entry = K_magnitude_entry / np.real(Psi_yy_entry)

    W_xx_initial = np.sqrt(2 / np.imag(Psi_xx_output[0]))
    W_xy_initial = np.sign(np.imag(Psi_xy_output[0])) * np.sqrt(
        2 / abs(np.imag(Psi_xy_output[0]))
    )
    W_yy_initial = np.sqrt(2 / np.imag(Psi_yy_output[0]))
    R_xx_initial = K_magnitude_array[0] / np.real(Psi_xx_output[0])
    R_xy_initial = K_magnitude_array[0] / np.real(Psi_xy_output[0])
    R_yy_initial = K_magnitude_array[0] / np.real(Psi_yy_output[0])

    numberOfPlotPoints = 25
    W_ellipse_points_entry = np.zeros([2, numberOfPlotPoints])
    W_ellipse_points_initial = np.zeros([2, numberOfPlotPoints])

    R_ellipse_points_entry = np.zeros([2, numberOfPlotPoints])
    R_ellipse_points_initial = np.zeros([2, numberOfPlotPoints])

    W_matrix_entry = np.array([[W_xx_entry, W_xy_entry], [W_xy_entry, W_yy_entry]])
    W_matrix_initial = np.array(
        [[W_xx_initial, W_xy_initial], [W_xy_initial, W_yy_initial]]
    )

    R_matrix_entry = np.array([[R_xx_entry, R_xy_entry], [R_xy_entry, R_yy_entry]])
    R_matrix_initial = np.array(
        [[R_xx_initial, R_xy_initial], [R_xy_initial, R_yy_initial]]
    )

    dummy_array = np.array(
        [
            np.cos(np.linspace(0, 2 * np.pi, numberOfPlotPoints)),
            np.sin(np.linspace(0, 2 * np.pi, numberOfPlotPoints)),
        ]
    )
    for ii in range(0, numberOfPlotPoints):
        W_ellipse_points_entry[:, ii] = np.matmul(W_matrix_entry, dummy_array[:, ii])
        W_ellipse_points_initial[:, ii] = np.matmul(
            W_matrix_initial, dummy_array[:, ii]
        )

        R_ellipse_points_entry[:, ii] = np.matmul(R_matrix_entry, dummy_array[:, ii])
        R_ellipse_points_initial[:, ii] = np.matmul(
            R_matrix_initial, dummy_array[:, ii]
        )

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(W_ellipse_points_entry[0, :], W_ellipse_points_entry[1, :], "r")
    plt.plot(W_ellipse_points_initial[0, :], W_ellipse_points_initial[1, :], "g")
    plt.plot(
        widths_entry[0] * widths_entry_eigvec[0, 0],
        widths_entry[0] * widths_entry_eigvec[0, 1],
        "ro",
    )
    plt.plot(
        widths_entry[1] * widths_entry_eigvec[1, 0],
        widths_entry[1] * widths_entry_eigvec[1, 1],
        "ro",
    )
    plt.axhline(y=0, color="k")
    plt.axvline(x=0, color="k")

    plt.subplot(1, 2, 2)
    plt.plot(R_ellipse_points_entry[0, :], R_ellipse_points_entry[1, :], "r")
    plt.plot(R_ellipse_points_initial[0, :], R_ellipse_points_initial[1, :], "g")
    plt.axhline(y=0, color="k")
    plt.axvline(x=0, color="k")


if __name__ == "__main__":
    plot()
