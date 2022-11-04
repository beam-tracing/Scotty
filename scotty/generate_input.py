#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Jul 24 14:31:24 2017

@author: Valerian Chen

Notes

- Bear in mind that Te currently uses the same function as ne. Not important for me since I don't care about temperature
- I've checked (CompareTopfiles) that B_z and B_r does indeed have the correct symmetry

Version history

v3 - Added a function for the generation of density (n_e) profiles.
     Used this function to make the profiles linear in rho rather than psi
v4 - Fixed bug with B_z and B_r
   - Made B_poloidal = B_poloidal max outside the last closed flux surface
   - Changed the way B_z, B_r, and B_poloidal are written, such that they use
     for loops. Not elegant, but at least I've made them work.
   - Cleaned up the transposes and the order part of reshape to be more understandable
v5 - B_toroidal now has correct units
v6 - Added two different methods of selecting the launch position
v7 - Fixed an issue where psi was transposed (shape(psi) = transpose(shape(B))) because meshgrid was using 'xy' by default, instead of ik
"""
import argparse
import pathlib

import numpy as np


def psi_fun(x_value, z_value, major_radius, minor_radius):
    """Gives the poloidal flux on the grid"""
    psi = ((x_value - major_radius) ** 2 + (z_value) ** 2) ** 0.5 / minor_radius
    return psi


def B_toroidal_fun(B_toroidal_max, x_array, z_array, major_radius):
    """Gives the toroidal B field on the grid"""
    B_toroidal = np.zeros([len(x_array), len(z_array)])
    for x_index in range(0, len(x_array)):
        x_value = x_array[x_index]
        B_toroidal[x_index, :] = B_toroidal_max * major_radius / x_value
    return B_toroidal


def B_poloidal_fun(B_poloidal_max, x_value, z_value, major_radius, minor_radius):
    """Gives the poloidal B field on the grid

    Magnetic axis assumed to be at

    - x = major_radus
    - z = 0
    """

    if ((x_value - major_radius) ** 2 + (z_value) ** 2) ** (0.5) > minor_radius:
        B_poloidal = B_poloidal_max
    else:
        B_poloidal = (
            B_poloidal_max
            * ((x_value - major_radius) ** 2 + (z_value) ** 2) ** 0.5
            / minor_radius
        )
    return B_poloidal


def B_r_fun(B_poloidal_max, x_array, z_array, major_radius, minor_radius):
    """Split B_poloidal into r and z components

    Magnetic axis assumed to be at

    - x = major_radus
    - z = 0
    """
    B_r = np.zeros([len(x_array), len(z_array)])

    for x_index in range(0, len(x_array)):
        for z_index in range(0, len(z_array)):
            x_value = x_array[x_index]
            z_value = z_array[z_index]
            if x_value == major_radius and z_value == 0:
                B_r[x_index, z_index] == 0
            else:
                B_poloidal = B_poloidal_fun(
                    B_poloidal_max, x_value, z_value, major_radius, minor_radius
                )
                B_r[x_index, z_index] = (
                    B_poloidal
                    * z_value
                    * ((x_value - major_radius) ** 2 + z_value**2) ** (-0.5)
                )
    return B_r


def B_z_fun(B_poloidal_max, x_array, z_array, major_radius, minor_radius):
    B_z = np.zeros([len(x_array), len(z_array)])

    for x_index in range(0, len(x_array)):
        for z_index in range(0, len(z_array)):
            x_value = x_array[x_index]
            z_value = z_array[z_index]
            if x_value == major_radius and z_value == 0:
                B_z[x_index, z_index] == 0
            else:
                B_poloidal = B_poloidal_fun(
                    B_poloidal_max, x_value, z_value, major_radius, minor_radius
                )
                B_z[x_index, z_index] = (
                    B_poloidal
                    * (x_value - major_radius)
                    * ((x_value - major_radius) ** 2 + z_value**2) ** (-0.5)
                )
    return B_z


def write_inbeam(
    minor_radius,
    major_radius,
    toroidal_launch_angle,
    poloidal_launch_angle,
    launch_position_x,
    launch_position_z,
    tau_step,
    torbeam_directory_path: pathlib.Path,
):
    """
    Writes inbeam.dat

    Refer to input-output.pdf for more detail.
    All offsets, flipping of signs, and so on should be done outside this function
    """
    inbeam_list = {
        "nabsroutine": 0,
        "nastra": 0,
        "nmaxh": 1,
        "noout": 0,
        "xzeff": 1,
        "ndns": 2,  # if =1, density profile is analytic
        "nte": 2,  # if =1, temperature profile is analytic
        "nmod": -1,  # O-mode (1), X-mode (-1)
        "ianexp": 2,  # if =1, magnetic equilibrium is analytic
        "nrel": 0,
        "ncdroutine": 2,
        "nrela": 0,
        "xf": 5.5e10,  # Launch freq (GHz)
        "xpoldeg": poloidal_launch_angle,  # poloidal launch angle
        "xtordeg": toroidal_launch_angle,  # toroidal launch angle
        "xxb": launch_position_x,  # x-coordinate of the launch position
        "xyb": 0,  # y-coordinate of the launch position
        "xzb": launch_position_z,  # z-coordinate of the launch position
        "xdns": 1.02e14,
        "edgdns": 1e13,
        "xe1": 2.0,
        "xe2": 0.5,
        "xte0": 25.0,
        "xteedg": 1.0,
        "xe1t": 2.0,
        "xe2t": 2.0,
        "xrtol": 1e-07,  # relative tolerance of the ODE solver
        "xatol": 1e-07,  # absolute tolerance of the ODE solver
        "xstep": tau_step,  # integration step in vacuum (cm)
        "xtbeg": 2.0,  # obsolete
        "xtend": 2.0,  # obsolete
        "xryyb": 50,  # beam radius of curvature (cm)
        "xrzzb": 50,
        "xwyyb": 5.0,  # beam width (cm)
        "xwzzb": 5.0,
        "xpw0": 1,  # Initial beam power (MW)
        "xrmaj": major_radius * 100.0,
        "xrmin": minor_radius * 100.0,
        "xb0": 2.01254,  # central toroidal magnetic field [T]
        "xdel0": 0.0,
        "xdeled": 0.0,
        "xelo0": 1.0,
        "xeloed": 1.5,
        "xq0": 1.0,
        "xqedg": 4.0,
        "nshot": 2,
        "rhostop": 1.2,
        "ncnstrn": 0,  # Neal Crocker's constraint
    }
    # inbeam_list += [' sgnm = 1'] # If 1, poloidal flux is minimum at the magnetic axis

    with open(torbeam_directory_path / "inbeam.dat", "w") as f:
        f.write("&edata\n")
        for key, value in inbeam_list.items():
            f.write(f" {key} = {value}\n")
        f.write("/")


def n_e_fun(nedata_psi, core_ne):
    nedata_length = len(nedata_psi)
    nedata_ne = np.zeros(nedata_length)

    nedata_ne = core_ne * (1 - nedata_psi**2)

    nedata_ne[-1] = 0.001
    return nedata_ne


def main(
    poloidal_launch_angle=0.0,
    toroidal_launch_angle=0.0,
    tau_step=0.05,
    B_toroidal_max=1.00,
    B_poloidal_max=0.0,
    core_ne=4.0,
    core_Te=0.01,
    aspect_ratio=1.5,
    minor_radius=0.5,
    torbeam_directory_path=pathlib.Path("."),
    nedata_length=101,
    Tedata_length=101,
):
    """Create TORBEAM input files

    Arguments
    ---------
    poloidal_launch_angle:
        In degrees
    toroidal_launch_angle:
        In degrees
    tau_step:
        Was 0.05, making it smaller to speed up runs
    B_toroidal_max:
        In Tesla
    B_poloidal_max:
        In Tesla
    core_ne:
        In 10^19 m-3 (IDL files, discussion w/ Jon)
    core_Te:

    aspect_ratio:
        major_radius/minor_radius
    minor_radius:
        In meters
    torbeam_directory_path:
        Directory to save output files
    nedata_length:

    Tedata_length:

    """

    major_radius = aspect_ratio * minor_radius
    launch_position_x = (major_radius + minor_radius) * 100 + 20.0
    launch_position_z = 0

    # Generate ne and Te
    nedata_psi = np.linspace(0, 1, nedata_length)
    Tedata_psi = np.linspace(0, 1, Tedata_length)

    nedata_ne = np.linspace(core_ne, 0, nedata_length)
    Tedata_Te = np.linspace(core_Te, 0, Tedata_length)

    nedata_ne = n_e_fun(nedata_psi, core_ne)
    Tedata_Te = n_e_fun(Tedata_psi, core_Te)

    nedata_ne[-1] = 0.001
    Tedata_Te[-1] = 0.001

    # Write ne and Te
    with open(torbeam_directory_path / "ne.dat", "w") as ne_data_file:
        ne_data_file.write(f"{int(nedata_length)}\n")
        for ii in range(0, nedata_length):
            ne_data_file.write("{:.8e} {:.8e} \n".format(nedata_psi[ii], nedata_ne[ii]))

    with open(torbeam_directory_path / "Te.dat", "w") as Te_data_file:
        Te_data_file.write(f"{int(Tedata_length)}\n")
        for ii in range(0, Tedata_length):
            Te_data_file.write("{:.8e} {:.8e} \n".format(Tedata_psi[ii], Tedata_Te[ii]))
    # --

    # Generate topfile variables
    buffer_factor = 1.1
    x_grid_length = 130
    x_grid_start = major_radius - buffer_factor * minor_radius  # in meters
    x_grid_end = major_radius + buffer_factor * minor_radius
    z_grid_length = 65  # Make sure this is a multiple of 5
    z_grid_start = -buffer_factor * minor_radius
    z_grid_end = buffer_factor * minor_radius

    x_grid = np.linspace(x_grid_start, x_grid_end, x_grid_length)
    z_grid = np.linspace(z_grid_start, z_grid_end, z_grid_length)

    B_r = np.zeros([x_grid_length, z_grid_length])
    B_z = np.zeros([x_grid_length, z_grid_length])
    B_t = np.zeros([x_grid_length, z_grid_length])
    psi = np.zeros([x_grid_length, z_grid_length])

    x_meshgrid, z_meshgrid = np.meshgrid(x_grid, z_grid, indexing="ij")
    B_t = B_toroidal_fun(B_toroidal_max, x_grid, z_grid, major_radius)
    B_r = B_r_fun(B_poloidal_max, x_grid, z_grid, major_radius, minor_radius)
    B_z = B_z_fun(B_poloidal_max, x_grid, z_grid, major_radius, minor_radius)
    psi = psi_fun(x_meshgrid, z_meshgrid, major_radius, minor_radius)
    print(np.shape(psi))
    print(np.shape(B_r))

    ## Check start
    # B_r_dataframe = pd.DataFrame(data=B_r, index=z_grid, columns=x_grid)
    #
    # plt.figure()
    # ax = sns.heatmap(B_r_dataframe, xticklabels=13,yticklabels=13, linewidths=0)
    # plt.xlabel('x / cm') # x-direction
    # plt.ylabel('z / cm')
    # ax.invert_yaxis()
    #
    # B_z_dataframe = pd.DataFrame(data=B_z, index=z_grid, columns=x_grid)
    #
    # plt.figure()
    # ax = sns.heatmap(B_z_dataframe, xticklabels=13,yticklabels=13, linewidths=0)
    # plt.xlabel('x / cm') # x-direction
    # plt.ylabel('z / cm')
    # ax.invert_yaxis()
    #
    # B_poloidal = (B_r**2 + B_z**2) ** (0.5)
    # B_poloidal_dataframe = pd.DataFrame(data=B_poloidal, index=z_grid, columns=x_grid)
    #
    # plt.figure()
    # ax = sns.heatmap(B_poloidal_dataframe, xticklabels=13,yticklabels=13, linewidths=0)
    # plt.xlabel('x / cm') # x-direction
    # plt.ylabel('z / cm')
    # ax.invert_yaxis()
    #
    # B_t_dataframe = pd.DataFrame(data=B_t, index=z_grid, columns=x_grid)
    #
    # plt.figure()
    # ax = sns.heatmap(B_t_dataframe, xticklabels=13,yticklabels=13, linewidths=0)
    # plt.xlabel('x / cm') # x-direction
    # plt.ylabel('z / cm')
    # ax.invert_yaxis()
    ## Check end

    # To enable writing in Torbeam's format
    # Reads and writes in C (and python) row col major
    # Transposes to get it to the correct major for Fortran
    z_grid = z_grid.reshape(z_grid_length // 5, 5, order="C")
    B_r = (np.transpose(B_r)).reshape(x_grid_length * z_grid_length // 5, 5, order="C")
    B_z = (np.transpose(B_z)).reshape(x_grid_length * z_grid_length // 5, 5, order="C")
    B_t = (np.transpose(B_t)).reshape(x_grid_length * z_grid_length // 5, 5, order="C")
    psi = (np.transpose(psi)).reshape(x_grid_length * z_grid_length // 5, 5, order="C")

    # Write topfile
    with open(torbeam_directory_path / "topfile", "w") as topfile_file:

        topfile_file.write("Dummy line\n")
        topfile_file.write(
            str(int(x_grid_length)) + " " + str(int(z_grid_length)) + "\n"
        )
        topfile_file.write("Dummy line\n")
        topfile_file.write("0 0 1\n")
        topfile_file.write("Grid: X-coordinates\n")
        for ii in range(0, x_grid_length):
            topfile_file.write("{:.8e}\n".format(x_grid[ii]))
        topfile_file.write("Grid: Z-coordinates\n")
        for ii in range(0, z_grid_length // 5):
            topfile_file.write(
                "{:.8}   {:.8}   {:.8}   {:.8}   {:.8} \n".format(
                    z_grid[ii, 0],
                    z_grid[ii, 1],
                    z_grid[ii, 2],
                    z_grid[ii, 3],
                    z_grid[ii, 4],
                )
            )
        topfile_file.write("Magnetic field: B_R\n")
        for ii in range(0, x_grid_length * z_grid_length // 5):
            topfile_file.write(
                "{:.8}   {:.8}   {:.8}   {:.8}   {:.8} \n".format(
                    B_r[ii, 0], B_r[ii, 1], B_r[ii, 2], B_r[ii, 3], B_r[ii, 4]
                )
            )
        topfile_file.write("Magnetic field: B_t\n")
        for ii in range(0, x_grid_length * z_grid_length // 5):
            topfile_file.write(
                "{:.8}   {:.8}   {:.8}   {:.8}   {:.8} \n".format(
                    B_t[ii, 0], B_t[ii, 1], B_t[ii, 2], B_t[ii, 3], B_t[ii, 4]
                )
            )
        topfile_file.write("Magnetic field: B_Z\n")
        for ii in range(0, x_grid_length * z_grid_length // 5):
            topfile_file.write(
                "{:.8}   {:.8}   {:.8}   {:.8}   {:.8} \n".format(
                    B_z[ii, 0], B_z[ii, 1], B_z[ii, 2], B_z[ii, 3], B_z[ii, 4]
                )
            )
        topfile_file.write("Poloidal flux: psi\n")
        for ii in range(0, x_grid_length * z_grid_length // 5):
            topfile_file.write(
                "{:.8}   {:.8}   {:.8}   {:.8}   {:.8} \n".format(
                    psi[ii, 0], psi[ii, 1], psi[ii, 2], psi[ii, 3], psi[ii, 4]
                )
            )

    write_inbeam(
        minor_radius,
        major_radius,
        toroidal_launch_angle,
        poloidal_launch_angle,
        launch_position_x,
        launch_position_z,
        tau_step,
        torbeam_directory_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Writes TORBEAM input files")
    parser.add_argument(
        "--poloidal_launch_angle", default=0.0, type=float, help="in degrees"
    )
    parser.add_argument(
        "--toroidal_launch_angle", default=0.0, type=float, help="in degrees"
    )
    parser.add_argument(
        "--tau_step",
        default=0.05,
        type=float,
        help="was 0.05, making it smaller to speed up runs",
    )
    parser.add_argument(
        "--B_toroidal_max", default=1.00, type=float, help="in Tesla (?)"
    )
    parser.add_argument("--B_poloidal_max", default=0.0, type=float, help="in Tesla")
    parser.add_argument(
        "--core_ne",
        default=4.0,
        type=float,
        help="in 10^19 m-3 (IDL files, discussion w/ Jon)",
    )
    parser.add_argument("--core_Te", default=0.01)
    parser.add_argument(
        "--aspect_ratio", default=1.5, type=float, help="major_radius/minor_radius"
    )
    parser.add_argument("--minor_radius", default=0.5, type=float, help="in meters")
    parser.add_argument("--torbeam_directory_path", default=pathlib.Path("."))
    parser.add_argument("--nedata_length", default=101, type=int)
    parser.add_argument("--Tedata_length", default=101, type=int)

    args = parser.parse_args()

    main(**vars(args))
