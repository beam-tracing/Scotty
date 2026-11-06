# -*- coding: utf-8 -*-
# Copyright 2018 - 2023, Valerian Hall-Chen and the Scotty contributors
# SPDX-License-Identifier: GPL-3.0

"""Which group of the output each variable is stored in"""

from typing import Dict, Tuple
import xarray as xr

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
    analysis: xr.Dataset,
    dbs_analysis: xr.Dataset,
) -> xr.DataTree:
    """The output's groups, from what the analysis functions make"""
    from_dbs, dbs = _split(dbs_analysis, BEAM_FROM_DBS)
    # `analysis` has the path lengths as variables, `dbs_analysis` as coordinates
    from_dbs = from_dbs.drop_vars([c for c in from_dbs.coords if c in analysis.data_vars])
    return xr.DataTree.from_dict(
        {
            "inputs": inputs,
            "solver": solver_output,
            "beam": xr.merge([analysis, from_dbs], join="exact", compat="override"),
            "diagnostics/dbs": dbs,
        }
    )


def from_output_tree(dt: xr.DataTree) -> Dict[str, xr.Dataset]:
    """What `to_output_tree` was given, by name"""
    from_dbs, analysis = _split(dt["beam"].to_dataset(), BEAM_FROM_DBS)
    dbs_analysis = xr.merge(
        [dt["diagnostics/dbs"].to_dataset(), from_dbs], join="exact", compat="override"
    )
    return {
        "inputs": dt["inputs"].to_dataset(),
        "solver_output": dt["solver"].to_dataset(),
        "analysis": analysis,
        "dbs_analysis": dbs_analysis,
    }
