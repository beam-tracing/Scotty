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
import pathlib

# For find_B if using efit files directly
from scotty.fun_CFD import find_dpolflux_dR, find_dpolflux_dZ
from scotty.density_fit import density_fit, DensityFitLike
from scotty._version import __version__

# Type hints
from typing import Optional, Union, Sequence
from scotty.typing import PathLike


def check_H_output(H_output):
    tol = 1e-2
    
    if max(H_output) > tol:
        """
        H_output should be zero along the entire ray
        
        If not close to zero, that means numerical errors are large
        """
        raise ValueError(
            f"`H_output` is too large! `H_output` = '{H_output}'"
        )    
    
    return None

def check_input(
    H_output
):

    check_H_output(H_output)
    
    return None
