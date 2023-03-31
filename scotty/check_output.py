# -*- coding: utf-8 -*-
# Copyright 2022 - 2023, Valerian Hall-Chen and the Scotty contributors
# SPDX-License-Identifier: GPL-3.0

"""Checks various quantities that `beam_me_up <scotty.beam_me_up.beam_me_up>` calculates"""

from __future__ import annotations
import numpy as np
from warnings import warn

from scotty.typing import FloatArray


def check_H_output(H_output: FloatArray) -> None:
    """H_output should be zero along the entire ray.

    If not close to zero, that means numerical errors are large
    """
    tol = 1e-2

    max_H = max(H_output)
    if max_H > tol:
        warn(
            f"WARNING: `H_output` is too large! `max H_output = {max_H}` but `tol = {tol}`"
        )
    elif max_H > 1.0:
        raise ValueError(f"`H_output` is unphysically large! `max H_output = {max_H}`")


def check_Psi(
    Psi_xg_output: FloatArray, Psi_yg_output: FloatArray, Psi_gg_output: FloatArray
) -> None:
    r"""
    By definition, we need:

    .. math::

        \begin{align}
        Im(\Psi \cdot \hat{\mathbf{g}}) &= 0 \\
        Re(\Psi \cdot \hat{\mathbf{g}}) &= \nabla_K H
        \end{align}
    """
    tol = 1e-3

    max_Psi = max(np.imag(Psi_xg_output))
    if max_Psi > tol:
        warn(
            f"WARNING: `Im(Psi_xg)` is too large! `max Im(Psi_xg) = {max_Psi}` but `tol = {tol}`"
        )


def check_output(H_output: FloatArray) -> None:
    check_H_output(H_output)
