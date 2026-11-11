# -*- coding: utf-8 -*-
# Copyright 2018 - 2026, Valerian Hall-Chen and the Scotty contributors
# SPDX-License-Identifier: GPL-3.0

r"""The antenna: everything about the beam before it reaches the plasma.

See :ref:`code-layout` in the documentation for what goes here and what goes
in `scotty.diagnostics`.

- `scotty.antenna.gaussian`: the launched Gaussian beam's :math:`\Psi_w`, and
  Gaussian beams in vacuum (waist, Rayleigh length, propagation)
- `scotty.antenna.horns`: horns, and the Gaussian beam each launches
- `scotty.antenna.lenses`: lenses and other ABCD optics, acting on
  :math:`\Psi_w`
"""
