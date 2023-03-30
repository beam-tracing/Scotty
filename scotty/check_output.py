# -*- coding: utf-8 -*-
"""
Created on 08 November 2022

@author: VH Hall-Chen
Valerian Hongjie Hall-Chen
valerian.chen@gmail.com

Checks various quantities that beam_me_up calculates
"""

from __future__ import annotations
import numpy as np
import warnings



def check_H_output(H_output):
    """H_output should be zero along the entire ray.

    If not close to zero, that means numerical errors are large
    """
    tol = 1e-2

    if max(H_output) > tol:
        warnings.warn(
            (
                f"WARNING: `H_output` is too large! `max H_output` = '{max(H_output)}' but 'tol' = '{tol}'"
            )
        )
    elif max(H_output) > 1.0:
        raise ValueError(
            f"`H_output` is unphysically large! `max H_output` = '{max(H_output)}'"
        )

    return None


def check_Psi(Psi_xg_output, Psi_yg_output, Psi_gg_output):
    # By definition, we need
    # Im(\Psi \cdot \hat{\mathbf{g}}) = 0
    # Re(\Psi \cdot \hat{\mathbf{g}}) = \nabla_K H

    tol = 1e-3

    if max(np.imag(Psi_xg_output)) > tol:
        warnings.warn(
            (
                f"WARNING: `H_output` is too large! `max Psi_xg` = '{max(Psi_xg_output)}' but 'tol' = '{tol}'"
            )
        )

    return None


def check_output(H_output):
    check_H_output(H_output)

    return None
