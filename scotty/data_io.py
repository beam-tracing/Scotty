import logging
from scotty.checks import Parameters, VALID_FIELDS
from typing import Tuple
import xarray as xr

log = logging.getLogger(__name__)

CYLINDRICAL_VECTOR_COMPONENTS = ["R", "zeta", "Z"]
CARTESIAN_VECTOR_COMPONENTS = ["X", "Y", "Z"]
SOLUTION_COMPONENTS = ["solution_1", "solution_2", "solution_3"]



def dict_to_dataset(data_dict: dict, data_coords: dict) -> xr.Dataset:
    return xr.Dataset(data_dict, data_coords)



def inputs_to_dicts(params: Parameters, field: VALID_FIELDS) -> Tuple[dict, dict]:

    cart = params.cartesian_flag

    inputs_dict = {
        # Main parameters
        "geometry": params.geometry,
        "poloidal_launch_angle_Torbeam": params.poloidal_launch_angle_deg_Torbeam,
        "toroidal_launch_angle_Torbeam": params.toroidal_launch_angle_deg_Torbeam,
        "launch_freq_GHz": params.launch_frequency_GHz,
        "launch_angular_frequency": params.launch_angular_frequency,
        "launch_wavenumber": params.launch_wavenumber,
        "launch_beam_width": params.launch_beam_width,
        "launch_beam_curvature": params.launch_beam_curvature,
        "launch_position": params.q_launch,
        "mode_flag": params.mode_flag_launch,

        # Solver settings and finite-difference parameters
        "benchmarking_flag": params.benchmarking_flag,
        "ray_tracing_flag": params.ray_tracing_flag,
        "launch_flag": params.launch_flag,
        "boundary_flag": params.boundary_flag,
        "relativistic_flag": params.relativistic_flag,
        "auto_delta_sign": params.auto_delta_sign,
        **({"delta_X": params.delta_X} if cart else {"delta_R": params.delta_R}),
        **({"delta_Y": params.delta_Y} if cart else {}),
        "delta_Z": params.delta_Z,
        **({"delta_K_X": params.delta_K_X} if cart else {"delta_K_R": params.delta_K_R}),
        **({"delta_K_Y": params.delta_K_Y} if cart else {"delta_K_zeta": params.delta_K_zeta}),
        "delta_K_Z": params.delta_K_Z,
        "len_tau": params.len_tau,
        "rtol": params.rtol,
        "atol": params.atol,
        "poloidal_flux_enter": params.poloidal_flux_enter,
        "poloidal_flux_zero_density": params.poloidal_flux_zero_density,
        "poloidal_flux_zero_temperature": params.poloidal_flux_zero_temperature,

        # Data input and output arguments
        "find_B_method": str(params.find_B_method),
        "magnetic_data_path": str(params.magnetic_data_path),
        "magnetic_data_interp_order": params.interp_order_magnetic_data_int,

        "ne_data_path": str(params.ne_data_path),
        "ne_data_interp_order": params.interp_order_ne_data, # int
        # "density_fit_parameters": params.density_fit_parameters, # TODO: netCDF cannot serialise (int, WindowsPath)
        "density_fit_method": params.density_fit_method,

        "Te_data_path": str(params.Te_data_path),
        "Te_data_interp_order": params.interp_order_Te_data,
        # "temperature_fit_parameters": params.temperature_fit_parameters, # TODO: netCDF cannot serialise (int, WindowsPath)
        "temperature_fit_method": params.temperature_fit_method,

        "interp_smoothing": params.interp_smoothing,
        "shot": params.shot,
        "equil_time": params.equil_time,
        
        "input_filename_suffix": params.input_filename_suffix,
        "output_path": str(params.output_path),
        "output_filename_suffix": params.output_filename_suffix,

        # Analysis and plotting arguments
        "figure_flag": params.figure_flag,
        "further_analysis_flag": params.further_analysis_flag,
        "detailed_analysis_flag": params.detailed_analysis_flag,
    }
    inputs_coords = {
        **({"X": field.X_coord} if cart else {"R": field.R_coord}), # type: ignore
        **({"Y": field.Y_coord} if cart else {}), # type: ignore
        "Z": field.Z_coord,
        "row": ["X","Y","Z"] if cart else ["R","zeta","Z"],
        "col": ["X","Y","Z"] if cart else ["R","zeta","Z"],
    }

    return inputs_dict, inputs_coords



def solver_output_to_dicts(params: Parameters) -> Tuple[dict, dict]:
    # put in docstring that K_zeta is toroidal mode number not actually the true toroidal component

    cart = params.cartesian_flag
    btf = not params.ray_tracing_flag
    solver_output_dict = {
        # launch stuff
        "q_launch_cartesian":   (["col_cart"], params.q_launch_cartesian),
        "q_launch_cylindrical": (["col_cyld"], params.q_launch_cylindrical),
        "K_launch_cartesian":   (["col_cart"], params.K_launch_cartesian),
        "K_launch_cylindrical": (["col_cyld"], params.K_launch_cylindrical),
        **({"Psi_3D_launch_labframe_cartesian":   (["row_cart", "col_cart"], params.Psi_3D_launch_labframe_cartesian)}   if btf else {}),
        **({"Psi_3D_launch_labframe_cylindrical": (["row_cyld", "col_cyld"], params.Psi_3D_launch_labframe_cylindrical)} if btf else {}),

        # entry stuff
        # for q and K, entry and initial are the same
        **({"Psi_3D_entry_labframe_cartesian":   (["row_cart", "col_cart"], params.Psi_3D_entry_labframe_cartesian)}   if btf else {}),
        **({"Psi_3D_entry_labframe_cylindrical": (["row_cyld", "col_cyld"], params.Psi_3D_entry_labframe_cylindrical)} if btf else {}),

        # initial stuff
        "q_initial_cartesian":   (["col_cart"], params.q_initial_cartesian),
        "q_initial_cylindrical": (["col_cyld"], params.q_launch_cylindrical),
        "K_initial_cartesian":   (["col_cart"], params.K_initial_cartesian),
        "K_initial_cylindrical": (["col_cyld"], params.K_initial_cylindrical),
        **({"Psi_3D_initial_labframe_cartesian":   (["row_cart", "col_cart"], params.Psi_3D_initial_labframe_cartesian)}   if btf else {}),
        **({"Psi_3D_initial_labframe_cylindrical": (["row_cyld", "col_cyld"], params.Psi_3D_initial_labframe_cylindrical)} if btf else {}),
        "e_hat_initial": (["col_cart"], params.e_hat_initial),
        "mode_flag_initial": params.mode_flag_initial,
        "mode_index": params.mode_index,
        "distance_from_launch_to_entry": params.distance_from_launch_to_entry,

        # general solver stuff
        "solver_status": params.solver_status,
        "solver_nfev": params.solver_nfev,
        "solver_duration": params.solver_duration,

        # ray-tracing solver output
        "q_X_output":    (["tau"], params.q_output_cartesian.T[0]),
        "q_Y_output":    (["tau"], params.q_output_cartesian.T[1]),
        "q_R_output":    (["tau"], params.q_output_cylindrical.T[0]),
        "q_zeta_output": (["tau"], params.q_output_cylindrical.T[1]),
        "q_Z_output":    (["tau"], params.q_output_cylindrical.T[2]),
        "q_output_cartesian":   (["tau", "row_cart"], params.q_output_cartesian),
        "q_output_cylindrical": (["tau", "row_cyld"], params.q_output_cylindrical),
        "K_X_output":    (["tau"], params.K_output_cartesian.T[0]),
        "K_Y_output":    (["tau"], params.K_output_cartesian.T[1]),
        "K_R_output":    (["tau"], params.K_output_cylindrical.T[0]),
        "K_zeta_output": (["tau"], params.K_output_cylindrical.T[1]),
        "K_Z_output":    (["tau"], params.K_output_cylindrical.T[2]),
        "K_output_cartesian":   (["tau", "col_cart"], params.K_output_cartesian),
        "K_output_cylindrical": (["tau", "col_cyld"], params.K_output_cylindrical),
        "K_output_magnitude": (["tau"], params.K_output_magnitude),

        # beam-tracing solver output
        **({"Psi_3D_output_labframe_cartesian":   (["tau", "row_cart", "col_cart"], params.Psi_3D_output_labframe_cartesian)}   if btf else {}),
        **({"Psi_3D_output_labframe_cylindrical": (["tau", "row_cyld", "col_cyld"], params.Psi_3D_output_labframe_cylindrical)} if btf else {}),
    }
    solver_output_coords = {
        "tau": params.tau_output,
        "row_cart": CARTESIAN_VECTOR_COMPONENTS,
        "row_cyld": CYLINDRICAL_VECTOR_COMPONENTS,
        "col_cart": CARTESIAN_VECTOR_COMPONENTS,
        "col_cyld": CYLINDRICAL_VECTOR_COMPONENTS,
    }

    return solver_output_dict, solver_output_coords