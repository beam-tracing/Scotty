import numpy as np
import pathlib
from scotty.geometry_3D import MagneticField_3D_Cartesian
from scotty.hamiltonian_3D import Hamiltonian_3D
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
    Psi_3D_lab_launch: FloatArray,
    Psi_3D_lab_entry: FloatArray,
    distance_from_launch_to_entry: float,
    vacuumLaunch_flag: bool,
    output_path: pathlib,
    output_filename_suffix: str,
    dH: Dict[str, ArrayLike]):

    X = solver_output.q_X
    Y = solver_output.q_Y
    Z = solver_output.q_Z
    K_X = solver_output.K_X
    K_Y = solver_output.K_Y
    K_Z = solver_output.K_Z
    polflux = field.polflux(X, Y, Z)

    tau_array = solver_output.tau
    numberOfDataPoints = len(tau_array)

    dH_dX = dH["dH_dX"]
    dH_dY = dH["dH_dX"]
    dH_dZ = dH["dH_dX"]
    dH_dKx = dH["dH_dKx"]
    dH_dKy = dH["dH_dKy"]
    dH_dKz = dH["dH_dKz"]

    # g = gradK of H
    g_magnitude = (dH_dKx**2 + dH_dKy**2 + dH_dKz**2) ** 0.5
    g_hat = ()