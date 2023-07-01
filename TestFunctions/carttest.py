#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 22:50:00 2023

@author: yvonne
Yvonne Ban
yban@hmc.edu

carttest (this file) is a file with methods to compute B field and bhat in Cartesian and cylindrical coordinates, and compare them respectively.
"""
#%%

import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
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
    find_dbhat_dR,
    find_dbhat_dZ,
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
        topfile_filename = magnetic_data_path + f"topfile{input_filename_suffix}.json"

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

#field = create_magnetic_geometry(
#    "torbeam",#find_B_method,
#    magnetic_data_path = "/Users/yvonne/Documents/GitHub/Scotty/data/",
#    input_filename_suffix = "_188839_1900ms",
#    interp_order = 5,
#    interp_smoothing = 0,
#    B_T_axis = None,
#    R_axis = None,
#    minor_radius_a = None,
#    B_p_a = None,
#    shot = None,
#    equil_time = None,
#    delta_R = -0.0001,
#    delta_Z = 0.0001,
#)

#%%
def B_xyz(mag_field_obj, q_R, q_zeta, q_Z):
    """
    Finds B-field in Cartesian (X, Y, Z) coordinates from MagneticField object.
    Input:
        mag_field_obj (MagneticField instance): instance of MagneticField object
        q_R (array): radius R of position q
        q_zeta (array): angle zeta of position q
        q_Z (array): height Z of position q
    Output:
        B_xyz array of B field in X, Y, Z coordinates, with coordinate dimensions as last dimension
    """
    B_R = mag_field_obj.B_R(q_R, q_Z)
    B_T = mag_field_obj.B_T(q_R, q_Z)
    B_Z = mag_field_obj.B_Z(q_R, q_Z)
    #B_mag = np.sqrt(B_R**2 + B_T**2 + B_Z**2)
    
    return np.stack((B_R*np.cos(q_zeta) - B_T*np.sin(q_zeta), B_R*np.sin(q_zeta) + B_T*np.cos(q_zeta), B_Z), axis=-1)

def grad_B_xyz(mag_field_obj, q_R, q_zeta, q_Z, delta=1e-3):
    """
    Finds gradient of B-field in Cartesian (X, Y, Z) coordinates from MagneticField object.
    Input:
        mag_field_obj (MagneticField instance): instance of MagneticField object
        q_R (array): radius R of position q
        q_zeta (array): angle zeta of position q
        q_Z (array): height Z of position q
        delta (float): increment over which to calculate gradient
    Output:
        grad_B_xyz array of B field in X, Y, Z coordinates, with coordinate dimensions as last dimension
    """
    #delta = 1e-3
    dx = delta
    dy = delta
    dz = delta
    
    zeta_x = np.arccos((q_R*np.cos(q_Z) + dx)/np.sqrt(q_R**2 + dx**2 + 2*dx*q_R*np.cos(q_zeta)))
    zeta_y = np.arcsin((q_R*np.sin(q_Z) + dy)/np.sqrt(q_R**2 + dy**2 + 2*dy*q_R*np.sin(q_zeta)))
    dzeta_x = zeta_x - q_zeta
    dzeta_y = zeta_y - q_zeta
    dr_x = dx*np.cos(zeta_x)
    dr_y = dy*np.sin(zeta_y)
    
    B_xplus = B_xyz(mag_field_obj, q_R +dr_x, q_zeta +dzeta_x, q_Z)[:,:,:,0]
    B_xminus = B_xyz(mag_field_obj, q_R -dr_x, q_zeta -dzeta_x, q_Z)[:,:,:,0]
    B_yplus = B_xyz(mag_field_obj, q_R +dr_y, q_zeta +dzeta_y, q_Z)[:,:,:,1]
    B_yminus = B_xyz(mag_field_obj, q_R -dr_y, q_zeta -dzeta_y, q_Z)[:,:,:,1]
    B_zplus = B_xyz(mag_field_obj, q_R, q_zeta, q_Z +dz)[:,:,:,2]
    B_zminus = B_xyz(mag_field_obj, q_R, q_zeta, q_Z -dz)[:,:,:,2]
    
    return 0.5*np.stack((B_xplus - B_xminus, B_yplus - B_yminus, B_zplus - B_zminus), axis=-1)/delta

def bhat(mag_field_obj, q_R, q_Z):
    """
    Finds bhat (in cylindrical (R, zeta, Z)) coordinates from MagneticField object.
    Input:
        mag_field_obj (MagneticField instance): instance of MagneticField object
        q_R (array): radius R of position q
        q_Z (array): height Z of position q
    Output:
        bhat array of B field in R, zeta, Z coordinates, with coordinate dimensions as last dimension
    """
    B_R = mag_field_obj.B_R(q_R, q_Z)
    B_T = mag_field_obj.B_T(q_R, q_Z)
    B_Z = mag_field_obj.B_Z(q_R, q_Z)
    B_mag = np.sqrt(B_R**2 + B_T**2 + B_Z**2)
    
    return np.stack((B_R/B_mag, B_T/B_mag, B_Z/B_mag), axis=-1)

def bhat_xyz(mag_field_obj, q_R, q_zeta, q_Z):
    """
    Finds bhat in Cartesian (X, Y, Z) coordinates from MagneticField object.
    Input:
        mag_field_obj (MagneticField instance): instance of MagneticField object
        q_R (array): radius R of position q
        q_zeta (array): angle zeta of position q
        q_Z (array): height Z of position q
    Output:
        bhat_xyz array of B field in X, Y, Z coordinates, with coordinate dimensions as last dimension
    """
    B_R = mag_field_obj.B_R(q_R, q_Z)
    B_T = mag_field_obj.B_T(q_R, q_Z)
    B_Z = mag_field_obj.B_Z(q_R, q_Z)
    B_mag = np.sqrt(B_R**2 + B_T**2 + B_Z**2)
    
    return np.stack(((B_R*np.cos(q_zeta) - B_T*np.sin(q_zeta))/B_mag, (B_R*np.sin(q_zeta) + B_T*np.cos(q_zeta))/B_mag, B_Z/B_mag), axis=-1)

def grad_bhat_xyz(mag_field_obj, q_R, q_zeta, q_Z, delta=1e-3):
    """
    Finds gradient of bhat in Cartesian (X, Y, Z) coordinates from MagneticField object.
    Input:
        mag_field_obj (MagneticField instance): instance of MagneticField object
        q_R (array): radius R of position q
        q_zeta (array): angle zeta of position q
        q_Z (array): height Z of position q
        delta (float): increment over which to calculate gradient
    Output:
        grad_bhat_xyz array of B field in X, Y, Z coordinates, with coordinate dimensions as last dimension
    """
    #delta = 1e-3
    dx = delta
    dy = delta
    dz = delta
    
    zeta_x = np.arccos((q_R*np.cos(q_Z) + dx)/np.sqrt(q_R**2 + dx**2 + 2*dx*q_R*np.cos(q_zeta)))
    zeta_y = np.arcsin((q_R*np.sin(q_Z) + dy)/np.sqrt(q_R**2 + dy**2 + 2*dy*q_R*np.sin(q_zeta)))
    dzeta_x = zeta_x - q_zeta
    dzeta_y = zeta_y - q_zeta
    dr_x = dx*np.cos(zeta_x)
    dr_y = dy*np.sin(zeta_y)
    
    bhat_xplus = bhat_xyz(mag_field_obj, q_R +dr_x, q_zeta +dzeta_x, q_Z)[:,:,:,0]
    bhat_xminus = bhat_xyz(mag_field_obj, q_R -dr_x, q_zeta -dzeta_x, q_Z)[:,:,:,0]
    bhat_yplus = bhat_xyz(mag_field_obj, q_R +dr_y, q_zeta +dzeta_y, q_Z)[:,:,:,1]
    bhat_yminus = bhat_xyz(mag_field_obj, q_R -dr_y, q_zeta -dzeta_y, q_Z)[:,:,:,1]
    bhat_zplus = bhat_xyz(mag_field_obj, q_R, q_zeta, q_Z +dz)[:,:,:,2]
    bhat_zminus = bhat_xyz(mag_field_obj, q_R, q_zeta, q_Z -dz)[:,:,:,2]
    
    return 0.5*np.stack((bhat_xplus - bhat_xminus, bhat_yplus - bhat_yminus, bhat_zplus - bhat_zminus), axis=-1)/delta

def compare_grad_B(mag_field_obj, q_R, q_zeta, q_Z, delta=1e-3):
    """
    Compares gradient of B-field found in Cartesian (X, Y, Z) and cylindrical (R, zeta, Z) coordinates.
    Input:
        mag_field_obj (MagneticField instance): instance of MagneticField object
        q_R (array): radius R of position q
        q_zeta (array): angle zeta of position q
        q_Z (array): height Z of position q
        delta (float): increment over which to calculate gradient
    Output:
        compare_grad_B array of B field in CARTESIAN (X, Y, Z) coordinates, with coordinate dimensions as last dimension
    """
    grad_B_xyz_array = grad_B_xyz(mag_field_obj, q_R, q_zeta, q_Z, delta=delta)
    
    grad_B_R_array = 0.5*(mag_field_obj.B_R(q_R +delta, q_Z) - mag_field_obj.B_R(q_R -delta, q_Z))/delta
    #grad_B_T_array = 0#.5*(mag_field_obj.B_Z(q_R, q_Z) - mag_field_obj.B_Z(q_R, q_Z))/delta
    grad_B_Z_array = 0.5*(mag_field_obj.B_Z(q_R, q_Z +delta) - mag_field_obj.B_Z(q_R, q_Z -delta))/delta
    
    return grad_B_xyz_array - np.stack((grad_B_R_array*np.cos(q_zeta), grad_B_R_array*np.sin(q_zeta), grad_B_Z_array), axis=-1)

def compare_grad_bhat(mag_field_obj, q_R, q_zeta, q_Z, delta=1e-3):
    """
    Compares gradient of bhat found in Cartesian (X, Y, Z) and cylindrical (R, zeta, Z) coordinates.
    Input:
        mag_field_obj (MagneticField instance): instance of MagneticField object
        q_R (array): radius R of position q
        q_zeta (array): angle zeta of position q
        q_Z (array): height Z of position q
        delta (float): increment over which to calculate gradient
    Output:
        compare_grad_bhat array of B field in CARTESIAN (X, Y, Z) coordinates, with coordinate dimensions as last dimension
    """
    grad_bhat_xyz_array = grad_bhat_xyz(mag_field_obj, q_R, q_zeta, q_Z, delta=delta)
    
    grad_bhat_R_array = 0.5*(bhat(mag_field_obj, q_R +delta, q_Z) - bhat(mag_field_obj, q_R -delta, q_Z))[:,:,:,0]/delta
    #grad_bhat_T_array = 0#.5*(bhat(mag_field_obj, q_R, q_Z) - bhat(mag_field_obj, q_R, q_Z))[:,:,:,1]/delta
    grad_bhat_Z_array = 0.5*(bhat(mag_field_obj, q_R, q_Z +delta) - bhat(mag_field_obj, q_R, q_Z -delta))[:,:,:,2]/delta
    
    return grad_bhat_xyz_array - np.stack((grad_bhat_R_array*np.cos(q_zeta), grad_bhat_R_array*np.sin(q_zeta), grad_bhat_Z_array), axis=-1)

#%%

data_path = '/Users/yvonne/Documents/GitHub/Scotty/results/'
plot_path = data_path

#analysis_output = np.load(data_path + 'analysis_output.npz')
#data_output = np.load(data_path + 'data_output.npz')
#data_input = np.load(data_path + 'data_input.npz')
#solver_output = np.load(data_path + 'solver_output.npz')

field = create_magnetic_geometry(
    "omfit",#find_B_method,
    magnetic_data_path = data_path,
    input_filename_suffix = '',
    interp_order = 5,
    interp_smoothing = None,
    B_T_axis = None,
    R_axis = None,
    minor_radius_a = None,
    B_p_a = None,
    shot = None,
    equil_time = None,
    delta_R = 1e-5,
    delta_Z = 1e-5,
)

#%%
# Compare values for B computed in Cartesian and cylindrical coordinates

#T_coord = np.linspace(0, 2*constants.pi)
#R_array, T_array, Z_array = np.meshgrid(field.R_coord, T_coord, field.Z_coord)

B_test = B_xyz(field, data_output['q_R_array'], data_output['q_zeta_array'], data_output['q_Z_array']) - np.stack((data_output['B_R_output']*np.cos(data_output['q_zeta_array']) - data_output['B_T_output']*np.sin(data_output['q_zeta_array']), data_output['B_R_output']*np.sin(data_output['q_zeta_array']) + data_output['B_T_output']*np.cos(data_output['q_zeta_array']), data_output['B_Z_output']), axis=-1)

print(np.max(B_test), np.min(B_test), np.sum(B_test))

#%%
# Compare values for bhat computed in Cartesian and cylindrical coordinates

bhat_test = bhat(field, data_output['q_R_array'], data_output['q_Z_array']) - np.stack((data_output['B_R_output']/data_output['B_magnitude'], data_output['B_T_output']/data_output['B_magnitude'], data_output['B_Z_output']/data_output['B_magnitude']), axis=-1)

print(np.max(bhat_test), np.min(bhat_test), np.sum(bhat_test))

bhat_xyz_test = bhat_xyz(field, data_output['q_R_array'], data_output['q_zeta_array'], data_output['q_Z_array']) - np.stack(((data_output['B_R_output']*np.cos(data_output['q_zeta_array']) - data_output['B_T_output']*np.sin(data_output['q_zeta_array']))/data_output['B_magnitude'], (data_output['B_R_output']*np.sin(data_output['q_zeta_array']) + data_output['B_T_output']*np.cos(data_output['q_zeta_array']))/data_output['B_magnitude'], data_output['B_Z_output']/data_output['B_magnitude']), axis=-1)

print(np.max(bhat_xyz_test), np.min(bhat_xyz_test), np.sum(bhat_xyz_test))

#%%
# Compare values for grad_B computed in Cartesian and cylindrical coordinates

#grad_B_test = compare_grad_B(field, data_output['q_R_array'], data_output['q_zeta_array'], data_output['q_Z_array'], delta=data_input['delta_R'])

delta = data_input['delta_R']
dx = delta
dy = delta
dz = delta

zeta_x = np.arccos((data_output['q_R_array']*np.cos(data_output['q_zeta_array']) + dx)/np.sqrt(data_output['q_R_array']**2 + dx**2 + 2*dx*data_output['q_R_array']*np.cos(data_output['q_zeta_array'])))
zeta_y = np.arcsin((data_output['q_R_array']*np.sin(data_output['q_zeta_array']) + dy)/np.sqrt(data_output['q_R_array']**2 + dy**2 + 2*dy*data_output['q_R_array']*np.sin(data_output['q_zeta_array'])))
dzeta_x = zeta_x - data_output['q_zeta_array']
dzeta_y = zeta_y - data_output['q_zeta_array']
dr_x = dx*np.cos(zeta_x)
dr_y = dy*np.sin(zeta_y)

B_xplus = B_xyz(field, data_output['q_R_array'] +dr_x, data_output['q_zeta_array'] +dzeta_x, data_output['q_Z_array'])[:,0]
B_xminus = B_xyz(field, data_output['q_R_array'] -dr_x, data_output['q_zeta_array'] -dzeta_x, data_output['q_Z_array'])[:,0]
B_yplus = B_xyz(field, data_output['q_R_array'] +dr_y, data_output['q_zeta_array'] +dzeta_y, data_output['q_Z_array'])[:,1]
B_yminus = B_xyz(field, data_output['q_R_array'] -dr_y, data_output['q_zeta_array'] -dzeta_y, data_output['q_Z_array'])[:,1]
B_zplus = B_xyz(field, data_output['q_R_array'], data_output['q_zeta_array'], data_output['q_Z_array'] +delta)[:,2]
B_zminus = B_xyz(field, data_output['q_R_array'], data_output['q_zeta_array'], data_output['q_Z_array'] -delta)[:,2]

grad_B_xyz_array = 0.5*np.stack((B_xplus - B_xminus, B_yplus - B_yminus, B_zplus - B_zminus), axis=-1)/delta

grad_B_R_array = 0.5*(field.B_R(data_output['q_R_array'] +delta, data_output['q_Z_array']) - field.B_R(data_output['q_R_array'] -delta, data_output['q_Z_array']))/delta
#grad_B_T_array = 0#.5*(field.B_Z(data_output['q_R_array'], data_output['q_Z_array']) - field.B_Z(data_output['q_R_array'], data_output['q_Z_array']))/delta
grad_B_Z_array = 0.5*(field.B_Z(data_output['q_R_array'], data_output['q_Z_array'] +delta) - field.B_Z(data_output['q_R_array'], data_output['q_Z_array'] -delta))/delta

grad_B_test = grad_B_xyz_array - np.stack((grad_B_R_array*np.cos(data_output['q_zeta_array']), grad_B_R_array*np.sin(data_output['q_zeta_array']), grad_B_Z_array), axis=-1)

#%%
# Plotting grad_B_test for clarity

plt.plot(grad_B_test[:,0],',', label='x')
plt.plot(grad_B_test[:,1],',', label='y')
plt.plot(grad_B_test[:,2],',', label='z')
plt.legend()
plt.title('Difference between grad_B_xyz and grad_B in cyl')
plt.savefig(plot_path + 'grad_B_test.png', bbox_inches='tight')

plt.plot(grad_B_test[:,0]/grad_B_xyz_array[:,0],',', label='x')
plt.plot(grad_B_test[:,1]/grad_B_xyz_array[:,1],',', label='y')
plt.plot(grad_B_test[:,2]/grad_B_xyz_array[:,2],',', label='z')
plt.legend()
plt.title('Difference between grad_B_xyz and grad_B in cyl\n(normalised to B_xyz)')
plt.savefig(plot_path + 'grad_B_test_norm.png', bbox_inches='tight')

#%%
# Compare values for grad_B computed in Cartesian and cylindrical coordinates

#grad_B_test = compare_grad_B(field, data_output['q_R_array'], data_output['q_zeta_array'], data_output['q_Z_array'], delta=data_input['delta_R'])

#%%
# Making synthetic data for tests

rmin=0
rmax=10
rsize=51
zmin=0
zmax=10
zsize=51
tmin=0
tmax=2*constants.pi
tsize=31

R = np.linspace(rmin,rmax,rsize)#gEQDSK['AuxQuantities']['R'].tolist()
Z = np.linspace(zmin,zmax,zsize)#gEQDSK['AuxQuantities']['Z'].tolist()
T = np.linspace(tmin,tmax,tsize)

Br = np.zeros((rsize,zsize))#gEQDSK['AuxQuantities']['Br'].tolist()
Bt = np.zeros((rsize,zsize))#gEQDSK['AuxQuantities']['Bt'].tolist()
Bz = np.ones((rsize,zsize))#gEQDSK['AuxQuantities']['Bz'].tolist()
Pol_Flux = np.zeros((rsize,zsize))


# else: x_coord is psi_n
#if x_type == "rho":
#    printw("! x_type is 'rho' based on profiles, will set pol_flux=(RHORZ)^2")
#    Pol_Flux = np.square(gEQDSK['AuxQuantities']['RHORZ']).tolist()
#elif x_type == "psi_n":
#    printw("! x_type is 'psi_n' based on profiles, will set pol_flux=PSIRZ_NORM")    Pol_Flux = gEQDSK['AuxQuantities']['PSIRZ_NORM'].tolist()
#else:
#    printe(f"! unknown x_type:{x_type} - ending")
#    OMFITx.End()

topfile_file = open(data_path+'topfile.json', 'w')
topfile_dict = {'R': R.tolist(), 'Z': Z.tolist(), 'Br': Br.tolist(), 'Bz': Bz.tolist(), 'Bt': Bt.tolist(), 'pol_flux': Pol_Flux.tolist()}
topfile_file.write(json.dumps(topfile_dict, indent=6))
topfile_file.close()

#%%
# Tests using synthetic data

Rarray, Zarray, Tarray = np.meshgrid(R,Z,T)

B_R_array = field.B_R(Rarray,Zarray)
B_T_array = field.B_T(Rarray,Zarray)
B_Z_array = field.B_Z(Rarray,Zarray)
B_mag_array = np.sqrt(B_R_array**2 + B_T_array**2 + B_Z_array**2)

# Sanity test: B_xyz works
B_test = B_xyz(field, Rarray, Tarray, Zarray) - np.stack((B_R_array*np.cos(Tarray) - B_T_array*np.sin(Tarray), B_R_array*np.sin(Tarray) + B_T_array*np.cos(Tarray), B_Z_array), axis=-1)

print(np.max(B_test), np.min(B_test), np.sum(B_test))

# Sanity test: bhat_xyz works
bhat_test = bhat(field, Rarray, Zarray) - np.stack((B_R_array/B_mag_array, B_T_array/B_mag_array, B_Z_array/B_mag_array), axis=-1)

print(np.max(bhat_test), np.min(bhat_test), np.sum(bhat_test))

bhat_xyz_test = bhat_xyz(field, Rarray, Tarray, Zarray) - np.stack(((B_R_array*np.cos(Tarray) - B_T_array*np.sin(Tarray))/B_mag_array, (B_R_array*np.sin(Tarray) + B_T_array*np.cos(Tarray))/B_mag_array, B_Z_array/B_mag_array), axis=-1)

print(np.max(bhat_xyz_test), np.min(bhat_xyz_test), np.sum(bhat_xyz_test))

# Test grad_B_xyz using compare_grad_B
grad_B_test = compare_grad_B(field, Rarray, Tarray, Zarray)
print(np.nanmax(grad_B_test), np.nanmin(grad_B_test), np.nansum(grad_B_test))

# Test grad_bhat_xyz using compare_grad_bhat
grad_bhat_test = compare_grad_bhat(field, Rarray, Tarray, Zarray)
print(np.nanmax(grad_bhat_test), np.nanmin(grad_bhat_test), np.nansum(grad_bhat_test))

