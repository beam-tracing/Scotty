# -*- coding: utf-8 -*-
"""
Created on Sat Sep 8 10:34:34 2024

@author: Aryan Law
"""

import numpy as np
import datatree
from scotty.plotting import (
    plot_poloidal_beam_path,
    plot_poloidal_crosssection,
    maybe_make_axis
    )
import matplotlib.pyplot as plt
from math import *
from scotty.fun_general import(
    find_widths_and_curvatures,
    freq_GHz_to_wavenumber,
    propagate_beam,
    find_K_lab_Cartesian
)
from scotty.analysis import(
    beam_width
)

import matplotlib.pyplot as plt
import scipy.constants as const
from scotty.geometry import MagneticField
from scipy.interpolate import UnivariateSpline

from scotty.beam_me_up import beam_me_up
import numpy as np

freq_sweep = np.linspace(30.0, 55.0, 15) #Range of frequencies to sweep
launch_beam_width = 0.04
launch_beam_curvature = -0.25
launch_position = np.array([2.278, 0, 0]) #Position of antenna

equil_time = 0.490
pol_flux_0 = 1.05496939**2
pol_flux_entry = 1.05451411**2



width1_at_launch = []
width2_at_launch = []
launch_width = []
R_Cutoff = []
Iteration = 0

radius_of_curvature_list = []
radius_of_circle_list = []
curvature_reflected_list = []

for freq in freq_sweep:
    Iteration = Iteration + 1
    print('Iteration')
    print(Iteration)

    launch_freq_GHz = freq

    
    kwargs_dict = {
        "poloidal_launch_angle_Torbeam": 0.0,
        "toroidal_launch_angle_Torbeam": 0.0,
        "launch_freq_GHz": launch_freq_GHz,
        "mode_flag": 1,
        "launch_beam_width": launch_beam_width,
        "launch_beam_curvature": launch_beam_curvature,
        "launch_position": launch_position,
        "density_fit_parameters": np.array([4.0, 1.0]),
        "delta_R": -0.00001,
        "delta_Z": 0.00001,
        'output_filename_suffix': 'CFS' + str(launch_freq_GHz),
        'output_path': "C:\\Users\\Aryan\\Documents\\Scotty\\CFS Output\\",
        "density_fit_method": "quadratic",
        "find_B_method": "analytical",
        "Psi_BC_flag": True,
        "figure_flag": True,
        "vacuum_propagation_flag": True,
        "vacuumLaunch_flag": True,
        "poloidal_flux_enter": 1.0,
        "poloidal_flux_zero_density": 1.0,
        "B_T_axis": 1.0,
        "B_p_a": 0.1,
        "R_axis": 1.5,
        "minor_radius_a": 0.5,
    }

    beam_me_up(**kwargs_dict)


    path = "C:\\Users\\Aryan\\Documents\\Scotty\\CFS Output\\scotty_output"  'CFS' + str(launch_freq_GHz) + ".h5"
    dt = datatree.open_datatree(path, engine="h5netcdf")

    charge = const.elementary_charge
    mass = const.electron_mass
    epsilon = const.epsilon_0

    
    for i in dt.analysis.keys():
        print(i)
    plot_poloidal_beam_path(dt,zoom=True)


    for i in dt.inputs.keys():
        print(i)
    

    field = MagneticField
    #Wave Properties
    cutoffindex = int(dt.analysis.cutoff_index)

    Psixx = np.array(dt.analysis.Psi_xx)
    Psixy = np.array(dt.analysis.Psi_xy)
    Psiyy = np.array(dt.analysis.Psi_yy)
    theta_m = np.array(dt.analysis.theta_m)
    theta = np.array(dt.analysis.theta)
    K_magnitude = np.array(dt.analysis.K_magnitude)
    q_R = np.array(dt.analysis.q_R)
    q_Z = np.array(dt.analysis.q_Z)
    K_zeta_initial = np.array(dt.analysis.K_zeta_initial)
    K_R = np.array(dt.analysis.K_R)
    K_Z = np.array(dt.analysis.K_Z)
    distance_along_line = np.array(dt.analysis.distance_along_line)
    Psi_3D_Cartesian = np.array(dt.analysis.Psi_3D_Cartesian)
    poloidal_flux = np.array(dt.analysis.poloidal_flux)

    R_Cutoff.append(q_R[cutoffindex-1])

    polflux = np.array(dt.inputs.poloidalFlux_grid.values)
    Rgrid =  np.array(dt.inputs.poloidalFlux_grid.coords['R'])
    Zgrid = np.array(dt.inputs.poloidalFlux_grid.coords['Z'])

    #plot all contour surfaces to be sure

    
    psi_select = poloidal_flux[cutoffindex-1]

    # Extract the contour line
    contour_collections =  plt.contour(Rgrid,Zgrid,np.transpose(polflux),levels = [psi_select],colors='orange')
    contour_paths = contour_collections.get_paths()
    contour = contour_paths[0]
    

    # Extract coordinates of the contour lines
    R_contour = contour.vertices[:, 0]
    z_contour = contour.vertices[:, 1]

    #adjust values accordingly
    R_filter_min = q_R[cutoffindex-1] - 0.01
    R_filter_max = q_R[cutoffindex-1] + 0.01
    Z_filter_max = q_Z[cutoffindex-1] + 0.5
    Z_filter_min = q_Z[cutoffindex-1] - 0.5

    radius_of_circle = (max(R_contour)-min(R_contour))/2

    R_cont_cleaned = []
    Z_cont_cleaned = []
    for i in range(len(R_contour)):
        if R_contour[i] < R_filter_max and R_contour[i] > R_filter_min and z_contour[i] < Z_filter_max and z_contour[i] > Z_filter_min:
            R_cont_cleaned.append(R_contour[i])
            Z_cont_cleaned.append(z_contour[i])
    
    R_half_cont = []
    Z_half_cont = []
    r0 = 1.5  # r-coordinate of the center
    for i in range(len(R_cont_cleaned)):
        if R_cont_cleaned[i]> r0:
            R_half_cont.append(R_cont_cleaned[i])
            Z_half_cont.append(Z_cont_cleaned[i])
    plt.plot(R_cont_cleaned,Z_cont_cleaned,'o')
    
    Z_half_cont.reverse()
    R_half_cont.reverse()

    plt.figure()
    plt.plot(Z_half_cont,R_half_cont,'o')
    
    spl = UnivariateSpline(Z_half_cont, R_half_cont)
    plt.plot(Z_half_cont,spl(Z_half_cont))
    #now, find derivatives

    first_deriv = spl.derivative(n=1)
    second_deriv = spl.derivative(n=2)

    measured_curvature = np.abs(second_deriv(q_Z[cutoffindex-1]))/(1+first_deriv(q_Z[cutoffindex-1])**2)**1.5

    radius_of_curvature = 1/measured_curvature

    radius_of_curvature_list.append(radius_of_curvature)
    radius_of_circle_list.append(radius_of_circle)

    widths, Psi_w_imag_eigvecs, curvatures, Psi_w_real_eigvecs = find_widths_and_curvatures(Psixx[cutoffindex-1], Psixy[cutoffindex-1], Psiyy[cutoffindex-1], K_magnitude[cutoffindex-1], theta_m[cutoffindex-1], theta[cutoffindex-1])

    #Beam Width
    def calculate_widths(Psixx,Psixy,Psiyy,distance_along_line):
        thresh_gradgrad = 100
        thresh_intersect = 100 
        Psixx = Psixx.imag
        Psixy = Psixy.imag
        Psiyy = Psiyy.imag
        Psi_w_imag = np.array([[Psixx, Psixy], [Psixy, Psiyy]])
        eigvals_im, eigvecs_im = np.linalg.eigh(np.moveaxis(Psi_w_imag, -1, 0))
        # Note the issue with function is that when the eigenvalues intersect, they might be switched around
        widths = np.sqrt(2 / eigvals_im)
        width1 = np.copy(widths[:, 0])
        width2 = np.copy(widths[:, 1])

        eigvec1_x = np.copy(eigvecs_im[:, 0, 0])
        eigvec1_y = np.copy(eigvecs_im[:, 0, 1])

        eigvec2_x = np.copy(eigvecs_im[:, 1, 0])
        eigvec2_y = np.copy(eigvecs_im[:, 1, 1])
        
        # Following code is to switch the eigenvalues/eigenvalues back
        gradgrad_width1 = np.gradient(np.gradient(width1, distance_along_line), distance_along_line)
        
        idx_switch_im = np.argmax(gradgrad_width1)
        
        if gradgrad_width1[idx_switch_im] > thresh_gradgrad * np.mean(gradgrad_width1):
            if abs(width1[idx_switch_im] - width2[idx_switch_im]) < np.mean(width1) / thresh_intersect:
                for i in range(idx_switch_im, len(width1)):
                    width1[i] = widths[i, 1]
                    width2[i] = widths[i, 0]
        return width1, width2

    width1, width2 = calculate_widths(Psixx,Psixy,Psiyy,distance_along_line)

    curvature_reflected = (-(-radius_of_curvature/2)**-1 + (curvatures)**-1)**-1
    curvature_reflected_list.append(curvature_reflected)

    wavenumber_K0 = freq_GHz_to_wavenumber(launch_freq_GHz)
    Psi_cutoff = Psi_3D_Cartesian[cutoffindex-1]
    toroidal_launch_angle_reflected = np.deg2rad(0)
    poloidal_launch_angle_reflected = np.deg2rad(180)

    Psi_3D_lab_launch_cartersian_reflected = np.zeros([3, 3], dtype="complex128")

    launch_position_reflected = np.array([q_R[cutoffindex-1], q_Z[cutoffindex-1], 0])

    Psi_3D_lab_launch_cartersian_reflected[0][0] = Psi_cutoff[0][0]
    Psi_3D_lab_launch_cartersian_reflected[0][1] = Psi_cutoff[0][1]
    Psi_3D_lab_launch_cartersian_reflected[0][2] = Psi_cutoff[0][2]
    Psi_3D_lab_launch_cartersian_reflected[1][0] = Psi_cutoff[1][0]
    Psi_3D_lab_launch_cartersian_reflected[2][0] = Psi_cutoff[2][0]


    Psi_3D_lab_launch_cartersian_reflected[1][1] = (-2*wavenumber_K0*cos(toroidal_launch_angle_reflected)/cos(poloidal_launch_angle_reflected))*(1/radius_of_curvature + ((tan(toroidal_launch_angle_reflected)*sin(poloidal_launch_angle_reflected))**2)/radius_of_curvature) + Psi_cutoff[1][1]

    Psi_3D_lab_launch_cartersian_reflected[2][1] = (-2*wavenumber_K0*tan(toroidal_launch_angle_reflected)*sin(poloidal_launch_angle_reflected))/radius_of_curvature + Psi_cutoff[2][1]

    Psi_3D_lab_launch_cartersian_reflected[2][2] = (-2*wavenumber_K0*cos(poloidal_launch_angle_reflected)/cos(toroidal_launch_angle_reflected))*(1/radius_of_curvature) + Psi_cutoff[2][2]


    Psi_3D_lab_launch_cartersian_reflected[1][2] = Psi_3D_lab_launch_cartersian_reflected[2][1]

    def find_K_lab(K_lab_Cartesian, q_lab_Cartesian):
        K_X = K_lab_Cartesian[0]
        K_Y = K_lab_Cartesian[1]
        K_Z = K_lab_Cartesian[2]

        q_R = q_lab_Cartesian[0]
        q_zeta = q_lab_Cartesian[1]
        q_Z = q_lab_Cartesian[2]

        K_lab = np.zeros(3)
        K_lab[0] = K_X * np.cos(q_zeta) + K_Y * np.sin(q_zeta)  # K_R
        K_lab[1] = (-K_X * np.sin(q_zeta) + K_Y * np.cos(q_zeta)) * q_R  # K_zeta
        K_lab[2] = K_Z
        return K_lab

    K_R_cartesian, K_zeta_cartesian, K_Z_cartesian = find_K_lab_Cartesian([K_R[cutoffindex-1], K_zeta_initial, K_Z[cutoffindex-1]], [q_R[cutoffindex-1], 0, 0])

    K_R_lab_reflected, K_zeta_lab_reflected, K_Z_lab_reflected = find_K_lab([-K_R_cartesian, K_zeta_cartesian, K_Z_cartesian], [q_R[cutoffindex-1], 0, 0])


    equil_time = 0.490
    kwargs_dict = {
        "poloidal_launch_angle_Torbeam": 0.0,
        "toroidal_launch_angle_Torbeam": 0.0,
        "launch_freq_GHz": launch_freq_GHz,
        "mode_flag": 1,
        "launch_beam_width": width1[cutoffindex-1],
        "launch_beam_curvature": curvature_reflected[0],
        "launch_position": np.array([q_R[cutoffindex-1], 0, 0]),
        "plasmaLaunch_Psi_3D_lab_Cartesian": Psi_3D_lab_launch_cartersian_reflected,
        "plasmaLaunch_K": np.array([K_R_lab_reflected,K_zeta_lab_reflected, K_Z_lab_reflected]),
        "density_fit_parameters": np.array([4.0, 1.0]),
        "delta_R": 0.00001,
        "delta_Z": 0.00001,
        'output_path' : "C:\\Users\\Aryan\\Documents\\Scotty\\CFS Reflected Output\\",
        'output_filename_suffix': 'CFS' + str(launch_freq_GHz),
        "density_fit_method": "quadratic",
        "find_B_method": "analytical",
        "Psi_BC_flag": True,
        "figure_flag": True,
        "vacuum_propagation_flag": False,
        "vacuumLaunch_flag": False,
        "auto_delta_sign": False,
        "poloidal_flux_enter": 1.0,
        "poloidal_flux_zero_density": 1.0,
        "B_T_axis": 1.0,
        "B_p_a": 0.1,
        "R_axis": 1.5,
        "minor_radius_a": 0.5,
    }

    beam_me_up(**kwargs_dict)


    path_reflected = "C:\\Users\\Aryan\\Documents\\Scotty\\CFS Reflected Output\\scotty_output"  + 'CFS' + str(launch_freq_GHz) + ".h5"
    dt_reflected = datatree.open_datatree(path_reflected, engine="h5netcdf")

    for i in dt_reflected.solver_output.keys():
        print(i)

    for i in dt_reflected.keys():
        print(i)

    for i in dt_reflected.analysis.keys():
        print(i)

    plot_poloidal_beam_path(dt_reflected,zoom=True)


    Psixx_reflected = np.array(dt_reflected.analysis.Psi_xx)
    Psixy_reflected = np.array(dt_reflected.analysis.Psi_xy)
    Psiyy_reflected = np.array(dt_reflected.analysis.Psi_yy)
    theta_m_reflected = np.array(dt_reflected.analysis.theta_m)
    theta_m_reflected = np.array(dt_reflected.analysis.theta)
    K_magnitude_reflected = np.array(dt_reflected.analysis.K_magnitude)
    q_R_reflected = np.array(dt_reflected.analysis.q_R)
    q_Z_reflected = np.array(dt_reflected.analysis.q_Z)
    K_zeta_initial_reflected = np.array(dt_reflected.analysis.K_zeta_initial)
    K_R_reflected = np.array(dt_reflected.analysis.K_R)
    K_Z_reflected = np.array(dt_reflected.analysis.K_Z)
    normalised_gyro_freqs = np.array(dt_reflected.analysis.normalised_gyro_freqs)
    Psi_3D_Cartesian = np.array(dt_reflected.analysis.Psi_3D_Cartesian)

    distance_along_line_reflected = np.array(dt_reflected.analysis.distance_along_line)

    width1_reflected, width2_reflected = calculate_widths(Psixx_reflected,Psixy_reflected,Psiyy_reflected,distance_along_line_reflected)

    Psi_w_reflected = np.zeros([2, 2], dtype="complex128")

    Psi_w_reflected = np.array([[Psixx_reflected[-1], Psixy_reflected[-1]], [Psixy_reflected[-1], Psiyy_reflected[-1]]])


    width1_reflected= list(width1_reflected)
    width2_reflected= list(width2_reflected)
    distance_along_line_reflected = list(distance_along_line_reflected)

    distance_to_launch = launch_position[0] - q_R_reflected[-1] 

    steps_to_launch = np.linspace(0, distance_to_launch, 100)


    
    for steps in steps_to_launch:
        psi_w_final_cartesian = propagate_beam(Psi_w_reflected, steps, launch_freq_GHz)
        beam_width_width1 = np.sqrt(2 / np.imag(psi_w_final_cartesian[0, 0]))
        width1_reflected.append(beam_width_width1)
        beam_width_width2 = np.sqrt(2 / np.imag(psi_w_final_cartesian[1, 1]))
        width2_reflected.append(beam_width_width2)
        distance_along_line_reflected.append(q_R_reflected[-1] + steps)


    width1_interpolate = UnivariateSpline(
        distance_along_line_reflected,
        width1_reflected,
        k=1,
        s=0
    )

    width2_interpolate = UnivariateSpline(
        distance_along_line_reflected,
        width2_reflected,
        k=1,
        s=0
    )
    
    width1_at_launch.append(width1_interpolate(launch_position[0] - q_R_reflected[0]))
    width2_at_launch.append(width2_interpolate(launch_position[0] - q_R_reflected[0]))

    launch_width.append(launch_beam_width)



    ax= None
    ax = maybe_make_axis(ax)

    plot_poloidal_crosssection(dt, ax=ax, highlight_LCFS=False)
    plot_poloidal_crosssection(dt_reflected, ax=ax, highlight_LCFS=False)
    launch_R = dt.inputs.launch_position.sel(col="R")
    launch_R_reflected = dt_reflected.inputs.launch_position.sel(col="R")
    launch_Z = dt.inputs.launch_position.sel(col="Z")
    launch_Z_reflected = dt_reflected.inputs.launch_position.sel(col="Z")
    ax.plot(
        np.concatenate([[launch_R], dt.analysis.q_R]),
        np.concatenate([[launch_Z], dt.analysis.q_Z]),
        ":k",
        label="Central (reference) ray",
    )
    ax.plot(
        np.concatenate([[launch_R_reflected], dt_reflected.analysis.q_R]),
        np.concatenate([[launch_Z_reflected], dt_reflected.analysis.q_Z]),
        ":k",
        label="Central (reference) ray (Reflected)",
    )

    width = beam_width(dt.analysis.g_hat, np.array([0.0, 1.0, 0.0]), dt.analysis.Psi_3D)
    beam_plus = dt.analysis.beam + width
    beam_minus = dt.analysis.beam - width
    ax.plot(beam_plus.sel(col="R"), beam_plus.sel(col="Z"), "--k")
    ax.plot(beam_minus.sel(col="R"), beam_minus.sel(col="Z"), "--k", label="Beam width")
    ax.scatter(launch_R, launch_Z, c="red", marker=">", label="Launch position")

    width_reflected = beam_width(dt_reflected.analysis.g_hat, np.array([0.0, 1.0, 0.0]), dt_reflected.analysis.Psi_3D)
    beam_plus_reflected = dt_reflected.analysis.beam + width_reflected
    beam_minus_reflected = dt_reflected.analysis.beam - width_reflected
    ax.plot(beam_plus_reflected.sel(col="R"), beam_plus_reflected.sel(col="Z"), "--k")
    ax.plot(beam_minus_reflected.sel(col="R"), beam_minus_reflected.sel(col="Z"), "--k", label="Beam width Reflected")
    ax.scatter(launch_R_reflected, launch_Z_reflected, c="red", marker=">", label="Launch position (reflected)")

    
    ## Write a wrapper function for this maybe
    R_max = max(beam_plus.sel(col="R").max(), beam_minus.sel(col="R").max(), beam_plus_reflected.sel(col="R").max(), beam_minus_reflected.sel(col="R").max())
    R_min = min(beam_plus.sel(col="R").max(), beam_minus.sel(col="R").max(), beam_plus_reflected.sel(col="R").max(), beam_minus_reflected.sel(col="R").max())
    Z_max = max(beam_plus.sel(col="Z").max(), beam_minus.sel(col="Z").max(), beam_plus_reflected.sel(col="Z").max(), beam_minus_reflected.sel(col="Z").max())
    Z_min = min(beam_plus.sel(col="Z").max(), beam_minus.sel(col="Z").max(), beam_plus_reflected.sel(col="Z").max(), beam_minus_reflected.sel(col="Z").max())

    buffer_R = 0.1 * (R_max - R_min)
    buffer_Z = 0.1 * (Z_max - Z_min)

    ax.set_xlim(R_min - buffer_R, R_max + buffer_R)
    ax.set_ylim(Z_min - buffer_Z, Z_max + buffer_Z)

    ax.legend()
    ax.set_title("Beam path (poloidal plane)")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    plt.close()
    

plt.figure()
plt.plot(R_Cutoff, radius_of_curvature_list)
plt.plot(R_Cutoff, radius_of_circle_list)
plt.ylabel('Radius of Curvature')
plt.xlabel('R Value')
plt.legend(["Measured Radius", "Actual Radius"])


plt.figure()
plt.plot(freq_sweep,width1_at_launch)
plt.plot(freq_sweep,width2_at_launch)
plt.plot(freq_sweep,launch_width)
plt.ylabel('Beam Width')
plt.xlabel('Frequency')
plt.legend(["Width1", "Width2", "Launch Width"])
plt.show()
