# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 12:06:45 2020

@author: VH Chen
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import pandas as pd    
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import chart_studio
from chart_studio import plotly as py
# pio.renderers.default = 'svg'
# pio.renderers.default = 'browser'

username = 'valerian.hall-chen' 
api_key = 'LLDO3Ig4bohBioZzIh9Y'
chart_studio.tools.set_credentials_file(username=username, api_key=api_key)

from Scotty_fun_general import find_widths_and_curvatures, find_q_lab_Cartesian, find_Psi_3D_lab_Cartesian, find_Psi_3D_lab, contract_special

def plot_beam_cross_section(Psi_3D_XYZ, x_hat_XYZ, y_hat_XYZ, numberOfPlotPoints=200):
    """
    Notes
    ----------
    The XYZ suffix indicates that the parameters should be in Cartesians. Do
    not confuse x,y (beam coordinates) for X,Y (lab coordinates)
    
    Parameters
    ----------
    Psi_3D_XYZ: ndarray (3,3)
        The beam parameter, \Psi
        
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
    
    plotAngles = np.linspace(0,2*np.pi,numberOfPlotPoints)
    plotWs = np.zeros_like(plotAngles)
    plotPhases = np.zeros_like(plotAngles)
    plotUVecs = np.zeros([numberOfPlotPoints,3])

    for ii, plotAngle in enumerate(plotAngles):
        plotUVec = x_hat_XYZ * np.cos(plotAngle) + y_hat_XYZ * np.sin(plotAngle)
        plotWs[ii] = np.sqrt(2 / np.dot(plotUVec,np.dot(np.imag(Psi_3D_XYZ),plotUVec)))
        plotPhases[ii] = (
                          np.dot(plotUVec,np.dot(np.real(Psi_3D_XYZ),plotUVec))
                          /
                          np.dot(plotUVec,np.dot(np.imag(Psi_3D_XYZ),plotUVec))
                          )
        plotUVecs[ii,:] = plotUVec
        
    # Psi_xx = np.dot(x_hat_XYZ,np.dot(Psi_3D_XYZ,x_hat_XYZ))
    # Psi_xy = np.dot(x_hat_XYZ,np.dot(Psi_3D_XYZ,y_hat_XYZ))
    # Psi_yy = np.dot(y_hat_XYZ,np.dot(Psi_3D_XYZ,y_hat_XYZ))        
        
    # plt.figure()
    # plt.scatter(plotWs*np.cos(plotAngles),plotWs*np.sin(plotAngles),
    #             c=plotPhases
    #             )
    # plt.colorbar()
    # plt.gca().set_aspect('equal', adjustable='box')
    
    
    return plotUVecs, plotWs


suffix = ''

loadfile = np.load('data_input' + suffix + '.npz')
launch_position = loadfile['launch_position']
launch_K = loadfile['launch_K']
loadfile.close()

loadfile = np.load('data_output' + suffix + '.npz')
g_magnitude_output = loadfile['g_magnitude_output']
q_R_array = loadfile['q_R_array']
q_zeta_array = loadfile['q_zeta_array']
q_Z_array = loadfile['q_Z_array']
K_R_array = loadfile['K_R_array']
K_zeta_initial = loadfile['K_zeta_initial']
K_Z_array = loadfile['K_Z_array']
b_hat_output = loadfile['b_hat_output']
g_hat_output = loadfile['g_hat_output']
x_hat_output = loadfile['x_hat_output']
y_hat_output = loadfile['y_hat_output']
Psi_3D_output = loadfile['Psi_3D_output']
Psi_3D_lab_launch = loadfile['Psi_3D_lab_launch']
dH_dKR_output = loadfile['dH_dKR_output']
dH_dKzeta_output = loadfile['dH_dKzeta_output']
dH_dKZ_output = loadfile['dH_dKZ_output']
loadfile.close()

loadfile = np.load('analysis_output' + suffix + '.npz')
Psi_3D_XYZ = loadfile['Psi_3D_Cartesian']
K_magnitude_array = loadfile['K_magnitude_array']
x_hats_XYZ = loadfile['x_hat_Cartesian']
y_hats_XYZ = loadfile['y_hat_Cartesian']
loadfile.close()




numberOfDataPoints = len(q_R_array)
numberOfPlotPoints = 17 # Best to choose 2^n +1
[q_X,q_Y,q_Z] = find_q_lab_Cartesian(np.array([q_R_array,q_zeta_array,q_Z_array]))

X_points = np.zeros([numberOfDataPoints,numberOfPlotPoints])
Y_points = np.zeros([numberOfDataPoints,numberOfPlotPoints])
Z_points = np.zeros([numberOfDataPoints,numberOfPlotPoints])



trace_ray = go.Scatter3d(
    x=q_X,
    y=q_Y,
    z=q_Z,
    mode='lines',
    line=dict(
        color='darkred',
        width=10
    )
    )

data_list = [trace_ray]


for ii in range(numberOfDataPoints):

    plotUVecs, plotWs = plot_beam_cross_section(Psi_3D_XYZ[ii,:,:], 
                            x_hats_XYZ[ii], y_hats_XYZ[ii],
                            numberOfPlotPoints=numberOfPlotPoints)

    # ax.scatter(q_X[ii]+plotUVecs[:,0],q_Y[ii]+plotUVecs[:,1],q_Z[ii]+plotUVecs[:,2],'k')



    X_points[ii,:] = q_X[ii] + plotWs * plotUVecs[:,0]
    Y_points[ii,:] = q_Y[ii] + plotWs * plotUVecs[:,1]
    Z_points[ii,:] = q_Z[ii] + plotWs * plotUVecs[:,2]
    
    width_line = go.Scatter3d(
        x=X_points[ii,:],
        y=Y_points[ii,:],
        z=Z_points[ii,:],
        mode='lines',
        line=dict(
            color='black',
            width=2
        )
        )    
    
    data_list = data_list + [width_line]
    
for jj in range(numberOfPlotPoints-1):
    if jj == 0:
        width_line = go.Scatter3d(
            x=X_points[:,jj],
            y=Y_points[:,jj],
            z=Z_points[:,jj],
            mode='lines',
            line=dict(
                color='darkblue',
                width=8
            )
            )    
    elif jj == 4:
        width_line = go.Scatter3d(
            x=X_points[:,jj],
            y=Y_points[:,jj],
            z=Z_points[:,jj],
            mode='lines',
            line=dict(
                color='darkgreen',
                width=8
            )
            )    
    else:
        width_line = go.Scatter3d(
            x=X_points[:,jj],
            y=Y_points[:,jj],
            z=Z_points[:,jj],
            mode='lines',
            line=dict(
                color='black',
                width=2
            )
            )    
    
    data_list = data_list + [width_line]

layout = go.Layout(
    margin = dict(
        l=10,
        r=10,
        b=10,
        t=10
        )
    )

fig = go.Figure(data=data_list, layout=layout)
# fig.show()
py.plot(fig, filename='test.html')