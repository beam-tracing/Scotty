# -*- coding: utf-8 -*-
# Copyright 2018 - 2023, Valerian Hall-Chen and the Scotty contributors
# SPDX-License-Identifier: GPL-3.0

"""The output: the datasets made from the inputs and the solver, and which group
of the output each variable is stored in"""

from __future__ import annotations

import logging
import numpy as np
from typing import Dict, Optional, Tuple, TYPE_CHECKING
import xarray as xr

# Only for type hints: scotty.checks imports scotty.compat, which imports this
if TYPE_CHECKING:
    from scotty.checks import Parameters, VALID_FIELDS

log = logging.getLogger(__name__)

CYLINDRICAL_VECTOR_COMPONENTS = ["R", "zeta", "Z"]
CARTESIAN_VECTOR_COMPONENTS = ["X", "Y", "Z"]
FIELD_ALIGNED_COMPONENTS = ["u1", "u2", "b"]
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
        "ray_tracing_flag": params.ray_tracing_flag,
        "launch_flag": params.launch_flag,
        # As a string, since netCDF cannot store `None`
        "Psi_BC_flag": str(params.Psi_BC_flag),
        "relativistic_flag": params.relativistic_flag,
        "first_harmonic_flag": params.first_harmonic_flag,
        "second_harmonic_flag": params.second_harmonic_flag,
        "upper_hybrid_flag": params.upper_hybrid_flag,
        "auto_delta_sign": params.auto_delta_sign,
        **({"delta_X": params.delta_X} if cart else {"delta_R": params.delta_R}),
        **({"delta_Y": params.delta_Y} if cart else {}),
        "delta_Z": params.delta_Z,
        **(
            {"delta_K_X": params.delta_K_X} if cart else {"delta_K_R": params.delta_K_R}
        ),
        **(
            {"delta_K_Y": params.delta_K_Y}
            if cart
            else {"delta_K_zeta": params.delta_K_zeta}
        ),
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
        "ne_data_interp_order": params.interp_order_ne_data,  # int
        # "density_fit_parameters": params.density_fit_parameters, # TODO: netCDF cannot serialise (int, WindowsPath)
        # As strings, since the methods can be ProfileFit objects, which
        # netCDF cannot serialise (as for `find_B_method` above)
        "density_fit_method": str(params.density_fit_method),
        "Te_data_path": str(params.Te_data_path),
        "Te_data_interp_order": params.interp_order_Te_data,
        # "temperature_fit_parameters": params.temperature_fit_parameters, # TODO: netCDF cannot serialise (int, WindowsPath)
        "temperature_fit_method": str(params.temperature_fit_method),
        "interp_smoothing": params.interp_smoothing,
        "shot": params.shot,
        "equil_time": params.equil_time,
        "input_filename_suffix": params.input_filename_suffix,
        "output_path": str(params.output_path),
        "output_filename_suffix": params.output_filename_suffix,
        # Plotting arguments
        "figure_flag": params.figure_flag,
        "analysis_flag": params.analysis_flag,
        "dbs_analysis_flag": params.dbs_analysis_flag,
    }
    # A plasma launch starts from the user's K and Psi (Cartesian, lab
    # frame), so keep them to make the run reproducible from its output
    if params.launch_flag == "plasma":
        inputs_dict["plasmaLaunch_K_cartesian"] = (
            ["col_cart"],
            np.asarray(params.K_plasmaLaunch_cartesian, dtype=float),
        )
        if not params.ray_tracing_flag:
            inputs_dict["plasmaLaunch_Psi_3D_lab_cartesian"] = (
                ["row_cart", "col_cart"],
                np.asarray(params.Psi_3D_plasmaLaunch_labframe_cartesian),
            )
    inputs_coords = {
        **({"X": field.X_coord} if cart else {"R": field.R_coord}),  # type: ignore
        **({"Y": field.Y_coord} if cart else {}),  # type: ignore
        "Z": field.Z_coord,
        "row": ["X", "Y", "Z"] if cart else ["R", "zeta", "Z"],
        "col": ["X", "Y", "Z"] if cart else ["R", "zeta", "Z"],
    }

    return inputs_dict, inputs_coords


def solver_output_to_dicts(params: Parameters) -> Tuple[dict, dict]:
    # put in docstring that K_zeta is toroidal mode number not actually the true toroidal component

    cart = params.cartesian_flag
    btf = not params.ray_tracing_flag
    solver_output_dict = {
        # launch stuff
        "q_launch_cartesian": (["col_cart"], params.q_launch_cartesian),
        "q_launch_cylindrical": (["col_cyld"], params.q_launch_cylindrical),
        "K_launch_cartesian": (["col_cart"], params.K_launch_cartesian),
        "K_launch_cylindrical": (["col_cyld"], params.K_launch_cylindrical),
        **(
            {
                "Psi_3D_launch_labframe_cartesian": (
                    ["row_cart", "col_cart"],
                    params.Psi_3D_launch_labframe_cartesian,
                )
            }
            if btf
            else {}
        ),
        **(
            {
                "Psi_3D_launch_labframe_cylindrical": (
                    ["row_cyld", "col_cyld"],
                    params.Psi_3D_launch_labframe_cylindrical,
                )
            }
            if btf
            else {}
        ),
        # entry stuff
        # for q and K, entry and initial are the same
        **(
            {
                "Psi_3D_entry_labframe_cartesian": (
                    ["row_cart", "col_cart"],
                    params.Psi_3D_entry_labframe_cartesian,
                )
            }
            if btf
            else {}
        ),
        **(
            {
                "Psi_3D_entry_labframe_cylindrical": (
                    ["row_cyld", "col_cyld"],
                    params.Psi_3D_entry_labframe_cylindrical,
                )
            }
            if btf
            else {}
        ),
        # initial stuff
        "q_initial_cartesian": (["col_cart"], params.q_initial_cartesian),
        "q_initial_cylindrical": (["col_cyld"], params.q_initial_cylindrical),
        "K_initial_cartesian": (["col_cart"], params.K_initial_cartesian),
        "K_initial_cylindrical": (["col_cyld"], params.K_initial_cylindrical),
        **(
            {
                "Psi_3D_initial_labframe_cartesian": (
                    ["row_cart", "col_cart"],
                    params.Psi_3D_initial_labframe_cartesian,
                )
            }
            if btf
            else {}
        ),
        **(
            {
                "Psi_3D_initial_labframe_cylindrical": (
                    ["row_cyld", "col_cyld"],
                    params.Psi_3D_initial_labframe_cylindrical,
                )
            }
            if btf
            else {}
        ),
        "e_hat_initial": (["col_field"], params.e_hat_initial),
        "mode_flag_initial": params.mode_flag_initial,
        "mode_index": params.mode_index,
        "distance_from_launch_to_entry": params.distance_from_launch_to_entry,
        # general solver stuff
        "solver_status": params.solver_status,
        "solver_nfev": params.solver_nfev,
        "solver_duration": params.solver_duration,
        # ray-tracing solver output
        "q_R_output": (["tau"], params.q_output_cylindrical.T[0]),
        "q_zeta_output": (["tau"], params.q_output_cylindrical.T[1]),
        "q_Z_output": (["tau"], params.q_output_cylindrical.T[2]),
        "q_output_cartesian": (["tau", "row_cart"], params.q_output_cartesian),
        "q_output_cylindrical": (["tau", "row_cyld"], params.q_output_cylindrical),
        "K_R_output": (["tau"], params.K_output_cylindrical.T[0]),
        "K_zeta_output": (["tau"], params.K_output_cylindrical.T[1]),
        "K_Z_output": (["tau"], params.K_output_cylindrical.T[2]),
        "K_output_cartesian": (["tau", "col_cart"], params.K_output_cartesian),
        "K_output_cylindrical": (["tau", "col_cyld"], params.K_output_cylindrical),
        "K_output_magnitude": (["tau"], params.K_output_magnitude),
        # beam-tracing solver output
        **(
            {
                "Psi_3D_output_labframe_cartesian": (
                    ["tau", "row_cart", "col_cart"],
                    params.Psi_3D_output_labframe_cartesian,
                )
            }
            if btf
            else {}
        ),
        **(
            {
                "Psi_3D_output_labframe_cylindrical": (
                    ["tau", "row_cyld", "col_cyld"],
                    params.Psi_3D_output_labframe_cylindrical,
                )
            }
            if btf
            else {}
        ),
    }
    solver_output_coords = {
        "tau": params.tau_output,
        "row_cart": CARTESIAN_VECTOR_COMPONENTS,
        "row_cyld": CYLINDRICAL_VECTOR_COMPONENTS,
        "col_cart": CARTESIAN_VECTOR_COMPONENTS,
        "col_cyld": CYLINDRICAL_VECTOR_COMPONENTS,
        "col_field": FIELD_ALIGNED_COMPONENTS,
    }

    return solver_output_dict, solver_output_coords


# The output's groups. The analysis functions make `analysis` and `dbs_analysis`;
# everything in `analysis` goes to `beam`, and everything in `dbs_analysis` goes
# to `diagnostics/dbs` except BEAM_FROM_DBS, which describe the beam rather than
# DBS and go to `beam`
BEAM_FROM_DBS = (
    "beam_waist",
    "binormal_hat",
    "cos_theta",
    "d_theta_d_tau",
    "det_Im_Psi_w",
    "k_perp1_hat",
    "normal_hat",
    "sin_theta",
    "theta",
    "xhat_dot_grad_bhat_dot_ghat",
    "xhat_dot_grad_bhat_dot_xhat",
    "xhat_dot_grad_bhat_dot_yhat",
    "yhat_dot_grad_bhat_dot_ghat",
    "yhat_dot_grad_bhat_dot_xhat",
    "yhat_dot_grad_bhat_dot_yhat",
)


def _split(ds: xr.Dataset, names: Tuple[str, ...]) -> Tuple[xr.Dataset, xr.Dataset]:
    """The variables of ``ds`` in ``names``, and the rest"""
    present = [name for name in names if name in ds.data_vars]
    return ds[present], ds.drop_vars(present)


def to_output_tree(
    inputs: xr.Dataset,
    solver_output: xr.Dataset,
    analysis: Optional[xr.Dataset] = None,
    dbs_analysis: Optional[xr.Dataset] = None,
) -> xr.DataTree:
    """The output's groups, from what the analysis functions make. Without
    ``analysis`` (``analysis_flag=False``) there is no ``beam`` group, and
    without ``dbs_analysis`` no ``diagnostics/dbs`` group"""
    groups = {"inputs": inputs, "solver": solver_output}
    if analysis is None:
        if dbs_analysis is not None:
            raise ValueError("The DBS analysis needs the beam analysis")
        return xr.DataTree.from_dict(groups)
    if dbs_analysis is None:
        groups["beam"] = analysis
        return xr.DataTree.from_dict(groups)
    from_dbs, dbs = _split(dbs_analysis, BEAM_FROM_DBS)
    # `analysis` has the path lengths as variables, `dbs_analysis` as coordinates
    from_dbs = from_dbs.drop_vars(
        [c for c in from_dbs.coords if c in analysis.data_vars]
    )
    groups["beam"] = xr.merge([analysis, from_dbs], join="exact", compat="override")
    groups["diagnostics/dbs"] = dbs
    return xr.DataTree.from_dict(groups)


def beam_group(dt: xr.DataTree) -> xr.DataTree:
    """The ``beam`` group of an output, or a ValueError saying why it is
    missing"""
    if "beam" not in dt.children:
        raise ValueError(
            "This output has no `beam` group: it was made with "
            "`analysis_flag=False`. Rerun with the analysis on"
        )
    return dt["beam"]


def from_output_tree(dt: xr.DataTree) -> Dict[str, xr.Dataset]:
    """What `to_output_tree` was given, by name. The analysis groups are
    absent from a run made with ``analysis_flag`` or ``dbs_analysis_flag``
    off, and so are their entries here"""
    groups = {
        "inputs": dt["inputs"].to_dataset(),
        "solver_output": dt["solver"].to_dataset(),
    }
    if "beam" not in dt.children:
        return groups
    beam = dt["beam"].to_dataset()
    if "diagnostics" not in dt.children or "dbs" not in dt["diagnostics"].children:
        groups["analysis"] = beam
        return groups
    from_dbs, analysis = _split(beam, BEAM_FROM_DBS)
    groups["analysis"] = analysis
    groups["dbs_analysis"] = xr.merge(
        [dt["diagnostics/dbs"].to_dataset(), from_dbs], join="exact", compat="override"
    )
    return groups
