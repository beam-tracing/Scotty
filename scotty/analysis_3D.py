import numpy as np
import pathlib
from scotty.analysis import set_vector_components_long_name
from scotty.check_output import check_output
from scotty.derivatives import derivative
from scotty.fun_general import (
    find_normalised_gyro_freq,
    find_normalised_plasma_freq,
    make_unit_vector_from_cross_product,
)
from scotty.geometry_3D import MagneticField_3D_Cartesian
from scotty.hamiltonian_3D import DielectricTensor_3D, Hamiltonian_3D
from scotty.profile_fit import ProfileFitLike
from scotty.typing import ArrayLike, FloatArray
from typing import Dict, Optional
import xarray as xr

CYLINDRICAL_VECTOR_COMPONENTS = ["R", "zeta", "Z"]
CARTESIAN_VECTOR_COMPONENTS = ["X", "Y", "Z"]



def immediate_analysis_3D(
    solver_output: xr.Dataset,
    field: MagneticField_3D_Cartesian,
    find_density_1D: ProfileFitLike,
    find_temperature_1D: Optional[ProfileFitLike],
    hamiltonian: Hamiltonian_3D,
    launch_angular_frequency: float,
    mode_flag: int,
    delta_X: float,
    delta_Y: float,
    delta_Z: float,
    delta_K_X: float,
    delta_K_Y: float,
    delta_K_Z: float,
    Psi_3D_lab_launch: FloatArray,        # Not used yet, so set vacuumLaunch_flag = False
    Psi_3D_lab_entry: FloatArray,         # Not used yet, so set vacuumLaunch_flag = False
    distance_from_launch_to_entry: float, # Not used yet, so set vacuumLaunch_flag = False
    vacuumLaunch_flag: bool,
    output_path: pathlib,
    output_filename_suffix: str,
    dH: Dict[str, ArrayLike]):

    q_X = solver_output.q_X
    q_Y = solver_output.q_Y
    q_Z = solver_output.q_Z
    dH_dX  = dH["dH_dX"]
    dH_dY  = dH["dH_dY"]
    dH_dZ  = dH["dH_dZ"]
    K_X = solver_output.K_X
    K_Y = solver_output.K_Y
    K_Z = solver_output.K_Z
    dH_dKx = dH["dH_dKx"]
    dH_dKy = dH["dH_dKy"]
    dH_dKz = dH["dH_dKz"]
    polflux = field.polflux(q_X, q_Y, q_Z)
    dpolflux_dX = field.d_polflux_dX(q_X, q_Y, q_Z, delta_X)
    dpolflux_dY = field.d_polflux_dY(q_X, q_Y, q_Z, delta_Y)
    dpolflux_dZ = field.d_polflux_dZ(q_X, q_Y, q_Z, delta_Z)
    tau_array = solver_output.tau
    numberOfDataPoints = len(tau_array)

    # Finding B field
    B_X = field.B_X(q_X, q_Y, q_Z)
    B_Y = field.B_X(q_X, q_Y, q_Z)
    B_Z = field.B_X(q_X, q_Y, q_Z)
    B_magnitude = field.magnitude(q_X, q_Y, q_Z)
    b_hat = field.unitvector(q_X, q_Y, q_Z)

    # Finding grad_bhat (dbhat/dX, dbhat/dY, dbhat/dZ)
    dbhat_dX = derivative(field.unitvector, dims="X", args={"X": q_X, "Y": q_Y, "Z": q_Z}, spacings=delta_X)
    dbhat_dY = derivative(field.unitvector, dims="Y", args={"X": q_X, "Y": q_Y, "Z": q_Z}, spacings=delta_Y)
    dbhat_dZ = derivative(field.unitvector, dims="Z", args={"X": q_X, "Y": q_Y, "Z": q_Z}, spacings=delta_Z)
    grad_bhat = np.zeros([numberOfDataPoints, 3, 3])
    grad_bhat[:, 0, :] = dbhat_dX
    grad_bhat[:, 1, :] = dbhat_dY
    grad_bhat[:, 2, :] = dbhat_dZ

    # Finding g = gradK_H and making x_hat, y_hat
    g_magnitude = np.sqrt(dH_dKx**2 + dH_dKy**2 + dH_dKz**2)
    g_hat = (np.block([[dH_dKx], [dH_dKy], [dH_dKz]]) / g_magnitude.data).T
    y_hat = make_unit_vector_from_cross_product(b_hat, g_hat)
    x_hat = make_unit_vector_from_cross_product(y_hat, g_hat)

    # There are two definitions of H used in Scotty:
    # H_Booker:  H is the determinant of the dispersion tensor D. Calculated using the Booker quartic
    # H_Cardano: H is the zero eigenvalue of the dispersion tensor D. Calculated using Cardano's formula
    # Here, we find H_Booker for both mode flags corresponding to 1 and -1
    H_Booker = hamiltonian(q_X, q_Y, q_Z, K_X, K_Y, K_Z)
    H_Booker_other = Hamiltonian_3D(field,
                                    launch_angular_frequency,
                                    -mode_flag,
                                    find_density_1D,
                                    delta_X,
                                    delta_Y,
                                    delta_Z,
                                    delta_K_X,
                                    delta_K_Y,
                                    delta_K_Z,
                                    find_temperature_1D)(q_X, q_Y, q_Z, K_X, K_Y, K_Z)
    
    # Calculating electron density, temperature, and epsilon
    electron_density = np.asfarray(find_density_1D(polflux))
    temperature = np.asfarray(find_temperature_1D(polflux)) if find_temperature_1D else None
    epsilon = DielectricTensor_3D(electron_density, launch_angular_frequency, B_magnitude, temperature)

    # Calculating (normalised) gyro freqs and (normalised) plasma freqs
    normalised_gyro_freqs = find_normalised_gyro_freq(electron_density, launch_angular_frequency, temperature)
    normalised_plasma_freqs = find_normalised_plasma_freq(B_magnitude, launch_angular_frequency, temperature)

    # Sanity check -- make sure the calculated quantities are reasonable
    check_output(H_Booker)
    print("The final value of H_Booker is", H_Booker[-1])

    df = xr.Dataset(
        {
            # Position and its derivatives
            "q_X":    (["tau"], q_X),
            "q_Y":    (["tau"], q_Y),
            "q_Z":    (["tau"], q_Z),
            "dH_dX":  (["tau"], dH_dX),
            "dH_dY":  (["tau"], dH_dY),
            "dH_dZ":  (["tau"], dH_dZ),

            # Wavevector and its derivatives
            "K_X":    (["tau"], K_X),
            "K_Y":    (["tau"], K_Y),
            "K_Z":    (["tau"], K_Z),
            "dH_dKx": (["tau"], dH_dKx),
            "dH_dKy": (["tau"], dH_dKy),
            "dH_dKz": (["tau"], dH_dKz),

            # Poloidal flux and its derivatives
            "polflux":     (["tau"], polflux),
            "dpolflux_dX": (["tau"], dpolflux_dX),
            "dpolflux_dY": (["tau"], dpolflux_dY),
            "dpolflux_dZ": (["tau"], dpolflux_dZ),

            # Magnetic field data
            "B_X": (["tau"], B_X),
            "B_Y": (["tau"], B_Y),
            "B_Z": (["tau"], B_Z),
            "B_magnitude": (["tau"], B_magnitude),
            "b_hat":     (["tau","col"], b_hat),
            "dbhat_dX":  (["tau","col"], dbhat_dX),
            "dbhat_dY":  (["tau","col"], dbhat_dY),
            "dbhat_dZ":  (["tau","col"], dbhat_dZ),
            "grad_bhat": (["tau","row","col"], grad_bhat),

            # Basis vectors
            "g_magnitude": (["tau"], g_magnitude),
            "g_hat": (["tau","col"], g_hat),
            "x_hat": (["tau","col"], x_hat),
            "y_hat": (["tau","col"], y_hat),

            # H booker stuff
            "H_Booker": (["tau"], H_Booker),
            "H_Booker_other": (["tau"], H_Booker_other),

            # Electron density and epsilon (e_bb, e_11, e_12)
            # Temperature is all the way below
            "electron_density": (["tau"], electron_density),
            "epsilon_para": (["tau"], epsilon.e_bb),
            "epsilon_bb":   (["tau"], epsilon.e_bb),
            "epsilon_perp": (["tau"], epsilon.e_11),
            "epsilon_11":   (["tau"], epsilon.e_11),
            "epsilon_g":    (["tau"], epsilon.e_12),
            "epsilon_12":   (["tau"], epsilon.e_12),

            # Normalised gyro frequencies and normalised plasma frequencies
            "normalised_gyro_freqs": (["tau"], normalised_gyro_freqs),
            "normalised_plasma_freqs": (["tau"], normalised_plasma_freqs),
        },
        coords = {
            "tau": tau_array,
            "row": CARTESIAN_VECTOR_COMPONENTS,
            "col": CARTESIAN_VECTOR_COMPONENTS,
            "q_X": q_X,
            "q_Y": q_Y,
            "q_Z": q_Z,
        },
    )

    if temperature is not None:
        df.update({"temperature": (["tau"], temperature)})
    
    if vacuumLaunch_flag:
        vacuum_only = {
            "Psi_3D_lab_entry": (["row", "col"], Psi_3D_lab_entry),
            "distance_from_launch_to_entry": distance_from_launch_to_entry,
        }
        df.update(vacuum_only)

    df.tau.attrs["long_name"] = "Parameterised distance along beam"
    set_vector_components_long_name(df)

    return df



def further_analysis_3D(
    inputs: xr.Dataset,
    df: xr.Dataset,
    Psi_3D_lab_entry_cartesian: FloatArray,
    output_path: Path,
    output_filename_suffix: str,
    field: MagneticField_3D_Cartesian,
    detailed_analysis_flag: bool,
    dH: Dict[str, ArrayLike]):

    # To continue

    return