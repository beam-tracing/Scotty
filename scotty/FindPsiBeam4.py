# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 12:06:45 2020

@author: VH Chen
"""


import numpy as np

from scotty.fun_general import find_q_lab_Cartesian


def plot_beam_cross_section(Psi_3D_XYZ, x_hat_XYZ, y_hat_XYZ, numberOfPlotPoints=200):
    r"""
    Notes
    ----------
    The XYZ suffix indicates that the parameters should be in Cartesians. Do
    not confuse x,y (beam coordinates) for X,Y (lab coordinates)

    Parameters
    ----------
    Psi_3D_XYZ: ndarray (3,3)
        The beam parameter, $\Psi$

    x_hat_XYZ: ndarray (3)
        x of the beam coordinates

    y_hat_XYZ: ndarray (3)
        y of the beam coordinates

    numberOfPlotPoints : Int, optional
        The default is 100.

    Returns
    -------
    None.

    """

    plotAngles = np.linspace(0, 2 * np.pi, numberOfPlotPoints)
    plotWs = np.zeros_like(plotAngles)
    plotPhases = np.zeros_like(plotAngles)
    plotUVecs = np.zeros([numberOfPlotPoints, 3])

    for ii, plotAngle in enumerate(plotAngles):
        plotUVec = x_hat_XYZ * np.cos(plotAngle) + y_hat_XYZ * np.sin(plotAngle)
        plotWs[ii] = np.sqrt(
            2 / np.dot(plotUVec, np.dot(np.imag(Psi_3D_XYZ), plotUVec))
        )
        plotPhases[ii] = np.dot(
            plotUVec, np.dot(np.real(Psi_3D_XYZ), plotUVec)
        ) / np.dot(plotUVec, np.dot(np.imag(Psi_3D_XYZ), plotUVec))
        plotUVecs[ii, :] = plotUVec

    return plotUVecs, plotWs


def plot(suffix: str = ""):
    import plotly.graph_objects as go
    import chart_studio
    from chart_studio import plotly as py

    username = "valerian.hall-chen"
    api_key = "LLDO3Ig4bohBioZzIh9Y"
    chart_studio.tools.set_credentials_file(username=username, api_key=api_key)

    loadfile = np.load("data_output" + suffix + ".npz")
    q_R_array = loadfile["q_R_array"]
    q_zeta_array = loadfile["q_zeta_array"]
    q_Z_array = loadfile["q_Z_array"]
    loadfile.close()

    loadfile = np.load("analysis_output" + suffix + ".npz")
    Psi_3D_XYZ = loadfile["Psi_3D_Cartesian"]
    x_hats_XYZ = loadfile["x_hat_Cartesian"]
    y_hats_XYZ = loadfile["y_hat_Cartesian"]
    loadfile.close()

    numberOfDataPoints = len(q_R_array)
    numberOfPlotPoints = 17  # Best to choose 2^n +1
    [q_X, q_Y, q_Z] = find_q_lab_Cartesian(
        np.array([q_R_array, q_zeta_array, q_Z_array])
    )

    X_points = np.zeros([numberOfDataPoints, numberOfPlotPoints])
    Y_points = np.zeros([numberOfDataPoints, numberOfPlotPoints])
    Z_points = np.zeros([numberOfDataPoints, numberOfPlotPoints])

    trace_ray = go.Scatter3d(
        x=q_X, y=q_Y, z=q_Z, mode="lines", line=dict(color="darkred", width=10)
    )

    data_list = [trace_ray]

    for ii in range(numberOfDataPoints):
        plotUVecs, plotWs = plot_beam_cross_section(
            Psi_3D_XYZ[ii, :, :],
            x_hats_XYZ[ii],
            y_hats_XYZ[ii],
            numberOfPlotPoints=numberOfPlotPoints,
        )

        X_points[ii, :] = q_X[ii] + plotWs * plotUVecs[:, 0]
        Y_points[ii, :] = q_Y[ii] + plotWs * plotUVecs[:, 1]
        Z_points[ii, :] = q_Z[ii] + plotWs * plotUVecs[:, 2]

        width_line = go.Scatter3d(
            x=X_points[ii, :],
            y=Y_points[ii, :],
            z=Z_points[ii, :],
            mode="lines",
            line=dict(color="black", width=2),
        )

        data_list = data_list + [width_line]

    for jj in range(numberOfPlotPoints - 1):
        if jj == 0:
            width_line = go.Scatter3d(
                x=X_points[:, jj],
                y=Y_points[:, jj],
                z=Z_points[:, jj],
                mode="lines",
                line=dict(color="darkblue", width=8),
            )
        elif jj == 4:
            width_line = go.Scatter3d(
                x=X_points[:, jj],
                y=Y_points[:, jj],
                z=Z_points[:, jj],
                mode="lines",
                line=dict(color="darkgreen", width=8),
            )
        else:
            width_line = go.Scatter3d(
                x=X_points[:, jj],
                y=Y_points[:, jj],
                z=Z_points[:, jj],
                mode="lines",
                line=dict(color="black", width=2),
            )

        data_list = data_list + [width_line]

    layout = go.Layout(margin=dict(l=10, r=10, b=10, t=10))

    fig = go.Figure(data=data_list, layout=layout)
    py.plot(fig, filename="test.html")


if __name__ == "__main__":
    plot()
