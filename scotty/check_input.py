# -*- coding: utf-8 -*-
"""
Created on 08 November 2022

@author: VH Hall-Chen
Valerian Hongjie Hall-Chen
valerian.chen@gmail.com

Checks input arguments of beam_me_up
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


def check_mode_flag(mode_flag):

    if mode_flag not in [-1, 1]:
        """
        Mode flag should be either -1 (X-mode) or 1 (O-mode)
        """
        raise ValueError(
            f"Bad value for `mode_flag`! Expected either 1 or -1, got '{mode_flag}'"
        )

    return None


def check_input(mode_flag):

    check_mode_flag(mode_flag)

    return None
