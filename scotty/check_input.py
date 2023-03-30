# -*- coding: utf-8 -*-
"""Created on 08 November 2022

@author: VH Hall-Chen
Valerian Hongjie Hall-Chen
valerian.chen@gmail.com

Checks input arguments of `beam_me_up <scotty.beam_me_up.beam_me_up>`

"""

from __future__ import annotations


def check_mode_flag(mode_flag: int) -> None:
    """Mode flag should be either -1 (X-mode) or 1 (O-mode)"""

    if mode_flag not in [-1, 1]:
        raise ValueError(
            f"Bad value for `mode_flag`! Expected either 1 or -1, got '{mode_flag}'"
        )


def check_input(mode_flag: int) -> None:
    check_mode_flag(mode_flag)
