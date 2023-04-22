#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 22:50:00 2023

@author: yvonne
"""
import numpy as np
from scipy import constants
import json
import pathlib
from typing import Optional, Union, Sequence, cast
from scotty.geometry import (
    MagneticField,
    CircularCrossSectionField,
    ConstantCurrentDensityField,
    InterpolatedField,
    CurvySlabField,
    EFITField,
)
from scotty.torbeam import Torbeam
from scotty.fun_general import (
    find_nearest,
    contract_special,
    make_unit_vector_from_cross_product,
    find_x0,
    find_waist,
    freq_GHz_to_angular_frequency,
    angular_frequency_to_wavenumber,
    find_Psi_3D_lab,
)

def create_magnetic_geometry(
    find_B_method: Union[str, MagneticField],
    magnetic_data_path: Optional[pathlib.Path] = None,
    input_filename_suffix: str = "",
    interp_order: int = 5,
    interp_smoothing: int = 0,
    B_T_axis: Optional[float] = None,
    R_axis: Optional[float] = None,
    minor_radius_a: Optional[float] = None,
    B_p_a: Optional[float] = None,
    shot: Optional[int] = None,
    equil_time: Optional[float] = None,
    delta_R: Optional[float] = None,
    delta_Z: Optional[float] = None,
    **kwargs,
) -> MagneticField:
    """Create an object representing the magnetic field geometry"""

    if isinstance(find_B_method, MagneticField):
        return find_B_method

    def missing_arg(argument: str) -> str:
        return f"Missing '{argument}' for find_B_method='{find_B_method}'"

    # Analytical geometries

    if find_B_method == "analytical":
        print("Analytical constant current density geometry")
        if B_T_axis is None:
            raise ValueError(missing_arg("B_T_axis"))
        if R_axis is None:
            raise ValueError(missing_arg("R_axis"))
        if minor_radius_a is None:
            raise ValueError(missing_arg("minor_radius_a"))
        if B_p_a is None:
            raise ValueError(missing_arg("B_p_a"))

        return ConstantCurrentDensityField(B_T_axis, R_axis, minor_radius_a, B_p_a)

    if find_B_method == "curvy_slab":
        print("Analytical curvy slab geometry")
        if B_T_axis is None:
            raise ValueError(missing_arg("B_T_axis"))
        if R_axis is None:
            raise ValueError(missing_arg("R_axis"))
        return CurvySlabField(B_T_axis, R_axis)

    if find_B_method == "unit-tests":
        print("Analytical circular cross-section geometry")
        if B_T_axis is None:
            raise ValueError(missing_arg("B_T_axis"))
        if R_axis is None:
            raise ValueError(missing_arg("R_axis"))
        if minor_radius_a is None:
            raise ValueError(missing_arg("minor_radius_a"))
        if B_p_a is None:
            raise ValueError(missing_arg("B_p_a"))

        return CircularCrossSectionField(B_T_axis, R_axis, minor_radius_a, B_p_a)

    ########################################
    # Interpolated numerical geometries from file

    if magnetic_data_path is None:
        raise ValueError(missing_arg("magnetic_data_path"))

    if find_B_method == "torbeam":
        print("Using Torbeam input files for B and poloidal flux")

        # topfile
        # Others: inbeam.dat, Te.dat (not currently used in this code)
        topfile_filename = magnetic_data_path + f"topfile{input_filename_suffix}"
        torbeam = Torbeam.from_file(topfile_filename)

        return InterpolatedField(
            torbeam.R_grid,
            torbeam.Z_grid,
            torbeam.B_R,
            torbeam.B_T,
            torbeam.B_Z,
            torbeam.psi,
            interp_order,
            interp_smoothing,
        )

    elif find_B_method == "omfit":
        print("Using OMFIT JSON Torbeam file for B and poloidal flux")
        topfile_filename = magnetic_data_path / f"topfile{input_filename_suffix}.json"

        with open(topfile_filename) as f:
            data = json.load(f)

        data_R_coord = np.array(data["R"])
        data_Z_coord = np.array(data["Z"])

        def unflatten(array):
            """Convert from column-major (TORBEAM, Fortran) to row-major order (Scotty, Python)"""
            return np.asarray(array).reshape(len(data_Z_coord), len(data_R_coord)).T

        return InterpolatedField(
            data_R_coord,
            data_Z_coord,
            unflatten(data["Br"]),
            unflatten(data["Bt"]),
            unflatten(data["Bz"]),
            unflatten(data["pol_flux"]),
            interp_order,
            interp_smoothing,
        )

    if find_B_method == "test":
        # Works nicely with the new MAST-U UDA output
        filename = magnetic_data_path / f"{shot}_equilibrium_data.npz"
        with np.load(filename) as loadfile:
            time_EFIT = loadfile["time_EFIT"]
            t_idx = find_nearest(time_EFIT, equil_time)
            print("EFIT time", time_EFIT[t_idx])

            return InterpolatedField(
                R_grid=loadfile["R_EFIT"],
                Z_grid=loadfile["Z_EFIT"],
                psi=loadfile["poloidalFlux_grid"][t_idx, :, :],
                B_T=loadfile["Bphi_grid"][t_idx, :, :],
                B_R=loadfile["Br_grid"][t_idx, :, :],
                B_Z=loadfile["Bz_grid"][t_idx, :, :],
                interp_order=interp_order,
                interp_smoothing=interp_smoothing,
            )
    if find_B_method == "test_notime":
        with np.load(magnetic_data_path) as loadfile:
            return InterpolatedField(
                R_grid=loadfile["R_EFIT"],
                Z_grid=loadfile["Z_EFIT"],
                psi=loadfile["poloidalFlux_grid"],
                B_T=loadfile["Bphi_grid"],
                B_R=loadfile["Br_grid"],
                B_Z=loadfile["Bz_grid"],
                interp_order=interp_order,
                interp_smoothing=interp_smoothing,
            )

    ########################################
    # Interpolated numerical geometries from EFIT data

    if delta_R is None:
        raise ValueError(missing_arg("delta_R"))
    if delta_Z is None:
        raise ValueError(missing_arg("delta_Z"))

    if find_B_method == "UDA_saved" and shot is None:
        print("Using MAST(-U) saved UDA data")
        # If using this part of the code, magnetic_data_path needs to include the filename
        # Assuming only one time present in file
        with np.load(magnetic_data_path) as loadfile:
            return EFITField(
                R_grid=loadfile["R_EFIT"],
                Z_grid=loadfile["Z_EFIT"],
                rBphi=loadfile["rBphi"],
                psi_norm_2D=loadfile["poloidalFlux_grid"],
                psi_unnorm_axis=loadfile["poloidalFlux_unnormalised_axis"],
                psi_unnorm_boundary=loadfile["poloidalFlux_unnormalised_boundary"],
                delta_R=delta_R,
                delta_Z=delta_Z,
                interp_order=interp_order,
                interp_smoothing=interp_smoothing,
            )

    # Data files with multiple time-slices

    if equil_time is None:
        raise ValueError(missing_arg("equil_time"))

    if find_B_method == "EFITpp":
        print(
            "Using MSE-constrained EFIT++ output files directly for B and poloidal flux"
        )
        return EFITField.from_EFITpp(
            magnetic_data_path / "efitOut.nc",
            equil_time,
            delta_R,
            delta_Z,
            interp_order,
            interp_smoothing,
        )

    if find_B_method == "UDA_saved" and shot is not None and shot <= 30471:  # MAST
        print(f"Using MAST shot {shot} from saved UDA data")
        # 30471 is the last shot on MAST
        # data saved differently for MAST-U shots
        filename = magnetic_data_path / f"{shot}_equilibrium_data.npz"
        return EFITField.from_MAST_saved(
            filename,
            equil_time,
            delta_R,
            delta_Z,
            interp_order,
            interp_smoothing,
        )

    if find_B_method == "UDA_saved" and shot is not None and shot > 30471:  # MAST-U
        print(f"Using MAST-U shot {shot} from saved UDA data")
        # 30471 is the last shot on MAST
        # data saved differently for MAST-U shots
        filename = magnetic_data_path / f"{shot}_equilibrium_data.npz"
        return EFITField.from_MAST_U_saved(
            filename,
            equil_time,
            delta_R,
            delta_Z,
            interp_order,
            interp_smoothing,
        )

    raise ValueError(f"Invalid find_B_method '{find_B_method}'")

field = create_magnetic_geometry(
    "torbeam",#find_B_method,
    magnetic_data_path = "/Users/yvonne/Documents/GitHub/Scotty/data/",
    input_filename_suffix = "_188839_1900ms",
    interp_order = 5,
    interp_smoothing = 0,
    B_T_axis = None,
    R_axis = None,
    minor_radius_a = None,
    B_p_a = None,
    shot = None,
    equil_time = None,
    delta_R = -0.0001,
    delta_Z = 0.0001,
)

def B_xyz(mag_field_obj, q_R, q_Z):
    """
    Finds B-field in Cartesian (X, Y, Z) coordinates from MagneticField object.
    Input:
        mag_field_obj (MagneticField instance): instance of MagneticField object
        q_R (array): q_R array
        q_Z (array): q_Z array
    Output:
        B_xyz array of B field in X, Y, Z coordinates, with coordinate dimensions as last dimension
    """
    B_R = mag_field_obj.B_R(q_R, q_Z)
    B_T = mag_field_obj.B_T(q_R, q_Z)# assuming this is rad
    B_Z = mag_field_obj.B_Z(q_R, q_Z)
    #B_mag = np.sqrt(B_R**2 + B_T**2 + B_Z**2)
    return np.stack(B_R*np.cos(B_T), B_R*np.sin(B_T), B_Z, axis=-1)

def grad_B_xyz(mag_field_obj, q_R, q_zeta, q_Z):
    """
    Finds gradient of B-field in Cartesian (X, Y, Z) coordinates from MagneticField object.
    Input:
        mag_field_obj (MagneticField instance): instance of MagneticField object
        q_R (array): q_R array
        q_zeta (array): q_zeta array
        q_Z (array): q_Z array
    Output:
        grad_B_xyz array of B field in X, Y, Z coordinates, with coordinate dimensions as last dimension
    """
    d = 1e-3
    dr_x = d/np.cos(q_zeta)
    dr_y = d/np.sin(q_zeta)
    B_xplus = B_xyz(mag_field_obj, q_R+dr_x, q_Z)[:,:,0]
    B_xminus = B_xyz(mag_field_obj, q_R-dr_x, q_Z)[:,:,0]
    B_yplus = B_xyz(mag_field_obj, q_R+dr_y, q_Z)[:,:,1]
    B_yminus = B_xyz(mag_field_obj, q_R-dr_y, q_Z)[:,:,1]
    B_zplus = B_xyz(mag_field_obj, q_R, q_Z+d)[:,:,2]
    B_zminus = B_xyz(mag_field_obj, q_R, q_Z-d)[:,:,2]
    
    return np.stack(B_xplus - B_xminus, B_yplus - B_yminus, B_zplus - B_zminus, axis=-1)

def bhat_xyz(mag_field_obj, q_R, q_Z):
    """
    Finds bhat in Cartesian (X, Y, Z) coordinates from MagneticField object.
    Input:
        mag_field_obj (MagneticField instance): instance of MagneticField object
        q_R (array): q_R array
        q_Z (array): q_Z array
    Output:
        bhat_xyz array of B field in X, Y, Z coordinates, with coordinate dimensions as last dimension
    """
    B_R = mag_field_obj.B_R(q_R, q_Z)
    B_T = mag_field_obj.B_T(q_R, q_Z)# assuming this is rad
    B_Z = mag_field_obj.B_Z(q_R, q_Z)
    B_mag = np.sqrt(B_R**2 + B_T**2 + B_Z**2)
    return np.stack(B_R*np.cos(B_T), B_R*np.sin(B_T), B_Z, axis=-1)/B_mag

def grad_bhat_xyz(mag_field_obj, q_R, q_zeta, q_Z):
    """
    Finds gradient of bhat in Cartesian (X, Y, Z) coordinates from MagneticField object.
    Input:
        mag_field_obj (MagneticField instance): instance of MagneticField object
        q_R (array): q_R array
        q_zeta (array): q_zeta array
        q_Z (array): q_Z array
    Output:
        grad_bhat_xyz array of B field in X, Y, Z coordinates, with coordinate dimensions as last dimension
    """
    d = 1e-3
    dr_x = d/np.cos(q_zeta)
    dr_y = d/np.sin(q_zeta)
    bhat_xplus = bhat_xyz(mag_field_obj, q_R+dr_x, q_Z)[:,:,0]
    bhat_xminus = bhat_xyz(mag_field_obj, q_R-dr_x, q_Z)[:,:,0]
    bhat_yplus = bhat_xyz(mag_field_obj, q_R+dr_y, q_Z)[:,:,1]
    bhat_yminus = bhat_xyz(mag_field_obj, q_R-dr_y, q_Z)[:,:,1]
    bhat_zplus = bhat_xyz(mag_field_obj, q_R, q_Z+d)[:,:,2]
    bhat_zminus = bhat_xyz(mag_field_obj, q_R, q_Z-d)[:,:,2]
    
    return np.stack(bhat_xplus - bhat_xminus, bhat_yplus - bhat_yminus, bhat_zplus - bhat_zminus, axis=-1)