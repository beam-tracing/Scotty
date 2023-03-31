# -*- coding: utf-8 -*-
# Copyright 2021 - 2023, Valerian Hall-Chen and the Scotty contributors
# SPDX-License-Identifier: GPL-3.0
"""This module calculates the Gaussian beam properties based on the horn and lens.

Much of the theory is very nicely explained in Goldsmith's Quasioptics. Just
note that what he calls 'flare angle', I call 'semiflare angle'.

- Table 7.1 for ratios of beam widths to aperture radii
"""
from typing import Dict

import numpy as np
from scipy import constants


def inch2m(length_inches):
    """Converts inches to meters"""
    return length_inches * 25.4 / 1000


class Horn:
    def output_beam(self, freq=None):
        """Return the width and curvature at the mouth of the horn
        for a given frequency.

        Note that this is curvature, the reciprocal of radius of curvature
        """
        raise NotImplementedError


class Conical_Horn(Horn):
    """
    Properties:

    - smooth walled
    - circular cross-section

    Attributes
    ==========
    name:
        Horn name
    aperture_radius:
        Radius of aperture in metres
    semiflare_angle:
    """

    def __init__(self, name: str, aperture_radius: float, semiflare_angle: float):
        self.name = name
        self.aperture_radius = aperture_radius
        self.semiflare_angle = semiflare_angle

    @property
    def inverse_slant_length(self):
        return np.sin(self.semiflare_angle) / self.aperture_radius

    def __repr__(self):
        return (
            f"{self.__class__.__name__}('{self.name}', "
            f"aperture_radius={self.aperture_radius}, "
            f"semiflare_angle={self.semiflare_angle})"
        )


class GoldsmithSymmetricConicalHorn(Conical_Horn):
    """Uses Goldsmith's *Quasioptics* textbook"""

    def output_beam(self, freq_GHz=None):
        width = 0.76 * self.aperture_radius
        curvature = self.inverse_slant_length
        return width, curvature


class GoldsmithAsymmetricConicalHorn(Conical_Horn):
    """Uses Goldsmith's *Quasioptics* textbook"""

    def output_beam(self, freq_GHz=None):
        # w_E > w_H seems to be the opposite of what Jon measures, though
        # Should pass it through the optics and everything and see what
        # I'd expect him to have measured
        width_E = 0.88 * self.aperture_radius
        width_H = 0.64 * self.aperture_radius
        width = [width_E, width_H]
        curvature = self.inverse_slant_length
        return width, curvature


class SpeirsAsymmetricConicalHorn(Conical_Horn):
    """Uses David Speirs' CST simulation data"""

    def output_beam(self, freq_GHz):
        # Beam waist widths don't change very much, it seems
        # From horn simulation's electric field (35 GHz)
        # in meters
        w0_E = 0.012948291561049162
        w0_H = 0.01731096934084839

        # Calculates the Rayleigh range
        zR_E = np.pi * w0_E**2 * (freq_GHz * 1e9 / constants.c)
        zR_H = np.pi * w0_H**2 * (freq_GHz * 1e9 / constants.c)

        # Simple assumption
        # distance_from_w0 = self.aperture_radius / np.tan(self.semiflare_angle)
        # Simulations
        distance_from_w0_E = 0.12785440710967151  # From 35 GHz simulation
        distance_from_w0_H = 0.12406338776025922  # positive because we want curvature to be positive at aperture

        width_E = w0_E * np.sqrt(1 + (distance_from_w0_E / zR_E) ** 2)
        width_H = w0_H * np.sqrt(1 + (distance_from_w0_H / zR_H) ** 2)
        width = [width_E, width_H]

        curvature_E = distance_from_w0_E / (distance_from_w0_E**2 + zR_E**2)
        curvature_H = distance_from_w0_H / (distance_from_w0_H**2 + zR_H**2)
        curvature = [curvature_E, curvature_H]

        # new simulations, with large simulation area to extract the properties
        # width = np.array([0.029874, 0.033298])
        # curvature = np.array([1/0.15743, 1/0.13844])
        return width, curvature


class SpeirsSymmetricConicalHorn(Conical_Horn):
    """Uses average of David Speirs' CST simulation data"""

    def output_beam(self, freq_GHz):
        # Not yet implemented

        width_E = 0.88 * self.aperture_radius
        width_H = 0.64 * self.aperture_radius
        width = [width_E, width_H]
        curvature = self.inverse_slant_length

        return width, curvature


class Scalar_Horn(Horn):
    """Aperture/diffraction-limited horn"""

    def __init__(self, name: str, aperture_radius: float):
        self.name = name
        self.aperture_radius = aperture_radius

    def __repr__(self):
        return f"Scalar_Horn('{self.name}', aperture_radius={self.aperture_radius})"

    def output_beam(self, freq=None):
        width = 0.644 * self.aperture_radius
        curvature = 0

        return width, curvature


def _MAST_V_band_aperature_radius() -> float:
    FWHM_angle = 25  # deg
    far_field_divergence_angle = np.rad2deg(
        np.arctan(np.tan(np.deg2rad(FWHM_angle)) / np.sqrt(2 * np.log(2)))
    )
    # np.log gives the natural log, which is what I want
    # myHorn.far_field_divergence_angle = FWHM_angle / 1.18 % Estimate, from Goldsmith

    # I assume the horn is aperture-limited, so the width at the output is
    # independent of freq
    mid_band_freq = (75.0 + 50.0) * 10**9 / 2
    w_0 = constants.c / (
        np.pi * mid_band_freq * np.tan(np.deg2rad(far_field_divergence_angle))
    )
    # w_0 = 0.003855191833541916 # This is what the calculation gives me
    # w_0 = 0.004054462688018294 # This is what Neal gets
    return w_0 / 0.644


KNOWN_HORNS: Dict[str, Horn] = {
    "MAST_V_band": Scalar_Horn("MAST_V_band", _MAST_V_band_aperature_radius()),
    "MAST_Q_band": SpeirsAsymmetricConicalHorn(
        "MAST_Q_band",
        aperture_radius=inch2m(1.44 / 2),
        semiflare_angle=np.deg2rad(30.0 / 2),
    ),
}


def make_my_horn(name: str) -> Horn:
    """Create one of a pre-existing set of parameterised horns

    Arguments
    =========
    name:
        One of "MAST_V_band", "MAST_Q_band"
    """

    try:
        return KNOWN_HORNS[name]
    except KeyError:
        raise ValueError(f"Unknown horn '{name}', expected one of {KNOWN_HORNS.keys()}")
