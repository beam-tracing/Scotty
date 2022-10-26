# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 20:15:24 2019

@author: VH Chen
"""

import numpy as np
import matplotlib.pyplot as plt
from scotty.fun_general import find_q_lab_Cartesian, find_nearest, contract_special


def plot(suffix: str = ""):
    loadfile = np.load("data_output" + suffix + ".npz")
    q_R_array = loadfile["q_R_array"]
    q_zeta_array = loadfile["q_zeta_array"]
    q_Z_array = loadfile["q_Z_array"]
    Psi_3D_output = loadfile["Psi_3D_output"]
    g_hat_output = loadfile["g_hat_output"]
    loadfile.close()

    loadfile = np.load("analysis_output" + suffix + ".npz")
    Psi_xx_output = loadfile["Psi_xx_output"]
    Psi_xy_output = loadfile["Psi_xy_output"]
    Psi_yy_output = loadfile["Psi_yy_output"]
    M_xx_output = loadfile["M_xx_output"]
    M_xy_output = loadfile["M_xy_output"]
    M_yy_output = loadfile["M_yy_output"]
    cutoff_index = loadfile["cutoff_index"]
    K_magnitude_array = loadfile["K_magnitude_array"]
    poloidal_flux_on_midplane = loadfile["poloidal_flux_on_midplane"]
    R_midplane_points = loadfile["R_midplane_points"]
    distance_along_line = loadfile["distance_along_line"]
    loadfile.close()

    loadfile = np.load("data_input" + suffix + ".npz")
    poloidalFlux_grid = loadfile["poloidalFlux_grid"]
    data_R_coord = loadfile["data_R_coord"]
    data_Z_coord = loadfile["data_Z_coord"]
    launch_position = loadfile["launch_position"]
    loadfile.close()

    [q_X_array, q_Y_array, q_Z_array] = find_q_lab_Cartesian(
        np.array([q_R_array, q_zeta_array, q_Z_array])
    )
    numberOfDataPoints = np.size(q_R_array)

    """
    Beam and ray path
    """
    ## For plotting the plasma in the toroidal plane
    index_polmin = find_nearest(poloidal_flux_on_midplane, 0)
    R_polmin = R_midplane_points[index_polmin]
    R_outboard = R_midplane_points[
        find_nearest(poloidal_flux_on_midplane[index_polmin:], 1) + index_polmin
    ]
    index_local_polmax = find_nearest(poloidal_flux_on_midplane[0:index_polmin], 10)
    R_inboard = R_midplane_points[
        find_nearest(poloidal_flux_on_midplane[index_local_polmax:index_polmin], 1)
        + index_local_polmax
    ]
    zeta_plot = np.linspace(-np.pi, np.pi, 1001)
    circle_outboard = np.zeros([1001, 2])
    circle_polmin = np.zeros([1001, 2])
    circle_inboard = np.zeros([1001, 2])
    circle_outboard[:, 0], circle_outboard[:, 1], _ = find_q_lab_Cartesian(
        np.array(
            [R_outboard * np.ones_like(zeta_plot), zeta_plot, np.zeros_like(zeta_plot)]
        )
    )
    circle_polmin[:, 0], circle_polmin[:, 1], _ = find_q_lab_Cartesian(
        np.array(
            [R_polmin * np.ones_like(zeta_plot), zeta_plot, np.zeros_like(zeta_plot)]
        )
    )
    circle_inboard[:, 0], circle_inboard[:, 1], _ = find_q_lab_Cartesian(
        np.array(
            [R_inboard * np.ones_like(zeta_plot), zeta_plot, np.zeros_like(zeta_plot)]
        )
    )
    ##

    ## For plotting how the beam propagates from launch to entry
    launch_position_X, launch_position_Y, launch_position_Z = find_q_lab_Cartesian(
        launch_position
    )
    entry_position_X, entry_position_Y, entry_position_Z = find_q_lab_Cartesian(
        np.array([q_R_array[0], q_zeta_array[0], q_Z_array[0]])
    )
    ##

    ## For plotting the width in the RZ plane
    W_vec_RZ = np.cross(g_hat_output, np.array([0, 1, 0]))
    W_vec_RZ_magnitude = np.linalg.norm(W_vec_RZ, axis=1)
    W_uvec_RZ = np.zeros_like(W_vec_RZ)  # Unit vector
    W_uvec_RZ[:, 0] = W_vec_RZ[:, 0] / W_vec_RZ_magnitude
    W_uvec_RZ[:, 1] = W_vec_RZ[:, 1] / W_vec_RZ_magnitude
    W_uvec_RZ[:, 2] = W_vec_RZ[:, 2] / W_vec_RZ_magnitude
    width_RZ = np.sqrt(
        2
        / np.imag(contract_special(W_vec_RZ, contract_special(Psi_3D_output, W_vec_RZ)))
    )
    W_line_RZ_1_Rpoints = q_R_array + W_uvec_RZ[:, 0] * width_RZ
    W_line_RZ_1_Zpoints = q_Z_array + W_uvec_RZ[:, 2] * width_RZ
    W_line_RZ_2_Rpoints = q_R_array - W_uvec_RZ[:, 0] * width_RZ
    W_line_RZ_2_Zpoints = q_Z_array - W_uvec_RZ[:, 2] * width_RZ
    ##

    ## For plotting the width in the XY plane
    g_hat_Cartesian = np.zeros([numberOfDataPoints, 3])
    g_hat_Cartesian[:, 0] = g_hat_output[:, 0] * np.cos(q_zeta_array) - g_hat_output[
        :, 1
    ] * np.sin(q_zeta_array)
    g_hat_Cartesian[:, 1] = g_hat_output[:, 0] * np.sin(q_zeta_array) + g_hat_output[
        :, 1
    ] * np.cos(q_zeta_array)
    g_hat_Cartesian[:, 2] = g_hat_output[:, 2]

    W_vec_XY = np.cross(g_hat_output, np.array([0, 0, 1]))
    W_vec_XY_magnitude = np.linalg.norm(W_vec_XY, axis=1)
    W_vec_XY[:, 0] = W_vec_XY[:, 0] / W_vec_XY_magnitude
    W_vec_XY[:, 1] = W_vec_XY[:, 1] / W_vec_XY_magnitude
    W_vec_XY[:, 2] = W_vec_XY[:, 2] / W_vec_XY_magnitude
    width_XY = np.sqrt(
        2
        / np.imag(contract_special(W_vec_XY, contract_special(Psi_3D_output, W_vec_XY)))
    )
    W_line_XY_1_Xpoints = q_X_array + W_vec_XY[:, 0] * width_XY
    W_line_XY_1_Ypoints = q_Y_array + W_vec_XY[:, 1] * width_XY
    W_line_XY_2_Xpoints = q_X_array - W_vec_XY[:, 0] * width_XY
    W_line_XY_2_Ypoints = q_Y_array - W_vec_XY[:, 1] * width_XY
    ##

    plot_every_n_points = 1

    R_start_index = 60
    R_end_index = 100
    Z_start_index = 40
    Z_end_index = 80

    out_index = numberOfDataPoints
    l_lc = (
        distance_along_line[:out_index:plot_every_n_points]
        - distance_along_line[cutoff_index]
    )

    plt.figure(figsize=(5, 5))
    plt.title("Poloidal Plane")
    contour_levels = np.linspace(0, 1, 11)
    CS = plt.contour(
        data_R_coord[R_start_index:R_end_index],
        data_Z_coord[Z_start_index:Z_end_index],
        np.transpose(
            poloidalFlux_grid[R_start_index:R_end_index, Z_start_index:Z_end_index]
        ),
        contour_levels,
        vmin=0,
        vmax=1.2,
        cmap="inferno",
    )
    plt.clabel(
        CS, inline=True, fontsize=10, inline_spacing=1, fmt="%1.1f", use_clabeltext=True
    )  # Labels the flux surfaces
    # plt.xlim(1.0,1.8)
    # plt.ylim(-0.7,0.1)
    plt.plot(
        q_R_array[:out_index:plot_every_n_points],
        q_Z_array[:out_index:plot_every_n_points],
        "k",
        linewidth=3,
    )
    plt.plot(
        [launch_position[0], q_R_array[0]], [launch_position[2], q_Z_array[0]], ":k"
    )
    plt.plot(
        W_line_RZ_1_Rpoints[:out_index:plot_every_n_points],
        W_line_RZ_1_Zpoints[:out_index:plot_every_n_points],
        "k",
        linewidth=1,
    )
    plt.plot(
        W_line_RZ_2_Rpoints[:out_index:plot_every_n_points],
        W_line_RZ_2_Zpoints[:out_index:plot_every_n_points],
        "k",
        linewidth=1,
    )
    # plt.xlim(data_R_coord[0],data_R_coord[-1])
    # plt.ylim(data_Z_coord[0],data_Z_coord[-1])

    plt.xlabel("R / m")
    plt.ylabel("Z / m")
    # tikzplotlib.save("propagation_poloidal.tex")
    plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.savefig("propagation_poloidal.jpg", dpi=200)

    circle_start_index = 478
    circle_end_index = 568
    plt.figure(figsize=(5, 5))
    plt.title("Toroidal Plane")
    plt.plot(
        circle_outboard[circle_start_index:circle_end_index, 0],
        circle_outboard[circle_start_index:circle_end_index, 1],
        "orange",
    )
    plt.plot(
        q_X_array[:out_index:plot_every_n_points],
        q_Y_array[:out_index:plot_every_n_points],
        "k",
        linewidth=3,
    )
    plt.plot(
        [launch_position_X, entry_position_X],
        [launch_position_Y, entry_position_Y],
        ":k",
    )
    plt.plot(
        W_line_XY_1_Xpoints[:out_index:plot_every_n_points],
        W_line_XY_1_Ypoints[:out_index:plot_every_n_points],
        "k",
        linewidth=1,
    )
    plt.plot(
        W_line_XY_2_Xpoints[:out_index:plot_every_n_points],
        W_line_XY_2_Ypoints[:out_index:plot_every_n_points],
        "k",
        linewidth=1,
    )
    plt.xlabel("X / m")
    plt.ylabel("Y / m")
    plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.savefig("propagation_toroidal.jpg", dpi=200)

    Psi_w = np.zeros([numberOfDataPoints, 2, 2], dtype="complex128")
    Psi_w[:, 0, 0] = Psi_xx_output
    Psi_w[:, 1, 1] = Psi_yy_output
    Psi_w[:, 1, 0] = Psi_xy_output
    Psi_w[:, 0, 1] = Psi_w[:, 1, 0]

    Psi_imag_eigval, Psi_imag_eigvec = np.linalg.eig(np.imag(Psi_w))
    Psi_real_eigval, Psi_real_eigvec = np.linalg.eig(np.real(Psi_w))

    W_eigval = np.sqrt(2 / Psi_imag_eigval)
    curv_eigval = np.zeros_like(W_eigval)
    curv_eigval[:, 0] = Psi_real_eigval[:, 0] / K_magnitude_array
    curv_eigval[:, 1] = Psi_real_eigval[:, 1] / K_magnitude_array

    M_w = np.zeros([numberOfDataPoints, 2, 2], dtype="complex128")
    M_w[:, 0, 0] = M_xx_output
    M_w[:, 1, 1] = M_yy_output
    M_w[:, 1, 0] = M_xy_output
    M_w[:, 0, 1] = M_w[:, 1, 0]

    # M_imag_eigval,M_imag_eigvec = np.linalg.eig(np.imag(M_w))
    M_real_eigval, M_real_eigvec = np.linalg.eig(np.real(M_w))

    eff_curv_eigval = np.zeros_like(W_eigval)
    eff_curv_eigval[:, 0] = M_real_eigval[:, 0] / K_magnitude_array
    eff_curv_eigval[:, 1] = M_real_eigval[:, 1] / K_magnitude_array

    import tikzplotlib

    plt.figure()
    plt.plot(
        l_lc,
        np.maximum(
            W_eigval[:out_index:plot_every_n_points, 0],
            W_eigval[:out_index:plot_every_n_points, 1],
        ),
    )
    plt.plot(
        l_lc,
        np.minimum(
            W_eigval[:out_index:plot_every_n_points, 0],
            W_eigval[:out_index:plot_every_n_points, 1],
        ),
    )
    plt.xlabel("l / m")
    plt.ylabel("W / m")
    plt.axvline(0, c="k", linestyle="dashed")
    tikzplotlib.save("widths.tex")

    plt.figure()
    plt.plot(
        l_lc,
        np.maximum(
            curv_eigval[:out_index:plot_every_n_points, 0],
            curv_eigval[:out_index:plot_every_n_points, 1],
        ),
    )
    plt.plot(
        l_lc,
        np.minimum(
            curv_eigval[:out_index:plot_every_n_points, 0],
            curv_eigval[:out_index:plot_every_n_points, 1],
        ),
    )
    plt.plot(
        l_lc,
        np.maximum(
            eff_curv_eigval[:out_index:plot_every_n_points, 0],
            eff_curv_eigval[:out_index:plot_every_n_points, 1],
        ),
    )
    plt.plot(
        l_lc,
        np.minimum(
            eff_curv_eigval[:out_index:plot_every_n_points, 0],
            eff_curv_eigval[:out_index:plot_every_n_points, 1],
        ),
    )
    plt.axvline(0, c="k", linestyle="dashed")
    plt.xlabel("l / m")
    plt.ylabel("(1 / R_b) / m")
    tikzplotlib.save("curvatures.tex")


if __name__ == "__main__":
    plot()
