# -*- coding: utf-8 -*-
# Copyright 2022 - 2023, Valerian Hall-Chen and the Scotty contributors
# SPDX-License-Identifier: GPL-3.0

"""Checks input arguments of `beam_me_up <scotty.beam_me_up.beam_me_up>`"""

from __future__ import annotations


def check_mode_flag(mode_flag: int) -> None:
    """Mode flag should be either -1 (X-mode) or 1 (O-mode)"""

    if mode_flag not in [-1, 1]:
        raise ValueError(
            f"Bad value for `mode_flag`! Expected either 1 or -1, got '{mode_flag}'"
        )


def check_input(mode_flag: int) -> None:
    check_mode_flag(mode_flag)
