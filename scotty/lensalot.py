# -*- coding: utf-8 -*-
"""
lens-a-lot

Created on Sat Apr 17 16:23:01 2021

@author: VH Hall-Chen (valerian@hall-chen.com)

Transfer matrix (2x2):

(   r'   )     ( A   B )  (   r   )
(        )  =  (       )  (       )
( theta' )     ( C   D )  ( theta )

Consider a ray entering and leaving an optical system. It is distance r above
the system axis, entering at an angle theta. It leaves at a distance r', and 
an angle theta'
"""

import numpy as np
from scipy import constants


def make_my_lens(name: str, lens_type: str = "thin", focal_length=None):
    known_lens_types = ["thin", "thick", "hyperbolic"]
    if lens_type not in known_lens_types:
        raise ValueError(
            f"Unknown lens type '{lens_type}', expected one of {known_lens_types}"
        )

    if lens_type == "thin":
        lenses = {
            "MAST_V_band": Thin_Lens("MAST_V_band", 0.125),
            "MAST_Q_band": Thin_Lens("MAST_Q_band", 0.27),
            "DBS_UCLA_DIII-D_240": Thin_Lens("DBS_UCLA_DIII-D_240", focal_length),
            # Harmonic mean of front and back focal distances
            "DBS_UCLA_MAST-U": Thin_Lens("DBS_UCLA_MAST-U", 0.1477692307692308),
        }
        try:
            return lenses[name]
        except KeyError:
            return ValueError(
                f"Unknown thin lens '{name}', expected one of {lenses.keys()}"
            )

    elif lens_type == "thick":
        lenses = {
            "MAST_V_band": Thick_Lens(
                "MAST_V_band", focal_length=0.125, thickness=0.0633, ref_index=1.53
            ),
            "MAST_Q_band": Thick_Lens(
                "MAST_Q_band", focal_length=0.27, thickness=0.0338, ref_index=1.53
            ),
        }
        try:
            return lenses[name]
        except KeyError:
            return ValueError(
                f"Unknown thick lens '{name}', expected one of {lenses.keys()}"
            )

    elif lens_type == "hyperbolic":
        lenses = {
            "MAST_V_band": ABCD_Lens(
                "MAST_V_band",
                A=1.000000000000,
                B=41.347131119605e-3,  # convert
                C=-0.006016900790e3,
                D=0.751218414102,
            ),
            "MAST_Q_band": ABCD_Lens(
                "MAST_Q_band",
                A=1.000000000000,
                B=22.103343171495e-3,  # convert
                C=-0.003423517875e3,
                D=0.924328809548,
            ),
        }
        try:
            return lenses[name]
        except KeyError:
            return ValueError(
                f"Unknown thick lens '{name}', expected one of {lenses.keys()}"
            )


def _frequency_to_wavenumber(frequency_GHz: float) -> float:
    angular_frequency = 2 * np.pi * 1e9 * frequency_GHz
    return angular_frequency / constants.c


def _check_matrix_is_2x2(Psi_w_in):
    if Psi_w_in.shape != (2, 2):
        raise ValueError(
            f"`Psi_w_in` must be 2x2 matrix, actual shape: {Psi_w_in.shape}"
        )


class Lens:
    """Abstract base class for different lens approximations"""

    def __init__(self, name: str, focal_length: float):
        self.name = name
        self.focal_length = focal_length

    def __repr__(self):
        return f"Lens('{self.name}', focal_length={self.focal_length})"

    def output_beam(self, Psi_w_in, freq_GHz):
        """Return the refracted beam

        - Assumes Psi_w is diagonalised
        - Doesn't assume that the diagnoalised Re(Psi_w)'s components are equal
          to each other, that is, the curvatures are allowed to be 'elliptical'

        Arguments
        =========
        Psi_w_in:
            Incident beam as a 2x2 matrix
        """

        raise NotImplementedError


class Thin_Lens(Lens):
    """
    Thin lens approximation
    """

    def __init__(self, name, focal_length):
        super().__init__(name, focal_length)

    def output_beam(self, Psi_w_in, freq_GHz):
        _check_matrix_is_2x2(Psi_w_in)
        wavenumber = _frequency_to_wavenumber(freq_GHz)
        Psi_w_out = Psi_w_in - np.eye(2) * wavenumber / self.focal_length
        return Psi_w_out


class Thick_Lens(Lens):
    """
    inherits Lens
    """

    def __init__(self, name, focal_length, thickness, ref_index):
        # Calling init of parent class
        super().__init__(name, focal_length)
        self.thickness = thickness
        self.ref_index = ref_index

    def output_beam(self, Psi_w_in, freq_GHz):
        _check_matrix_is_2x2(Psi_w_in)
        wavenumber = _frequency_to_wavenumber(freq_GHz)

        T_over_N = self.thickness / self.ref_index
        Psi_w_out2 = np.zeros_like(Psi_w_in, dtype="complex128")

        Psi_w_in2 = np.conj(Psi_w_in)
        print("B", T_over_N)
        print("C", -1 / self.focal_length)
        print("D", (1 - T_over_N / self.focal_length))
        # T_over_N = 0
        for ii in range(0, 2):
            Psi_w_out2[ii, ii] = (
                wavenumber
                * (wavenumber + T_over_N * Psi_w_in2[ii, ii]) ** (-1)
                * (
                    -wavenumber / self.focal_length
                    + (1 - T_over_N / self.focal_length) * Psi_w_in2[ii, ii]
                )
            )
        Psi_w_out = np.conj(Psi_w_out2)

        return Psi_w_out


class ABCD_Lens(Lens):
    """
    inherits Lens
    """

    def __init__(self, name, A, B, C, D):
        self.name = name
        self.matrix_A = A
        self.matrix_B = B
        self.matrix_C = C
        self.matrix_D = D

    def output_beam(self, Psi_w_in, freq_GHz):
        _check_matrix_is_2x2(Psi_w_in)
        wavenumber = _frequency_to_wavenumber(freq_GHz)

        q_out = np.zeros_like(Psi_w_in, dtype="complex128")
        q_in = np.zeros_like(Psi_w_in, dtype="complex128")
        Psi_w_out = np.zeros_like(Psi_w_in, dtype="complex128")

        for ii in range(0, 2):
            q_in[ii, ii] = 1 / (np.conj(Psi_w_in[0, 0]) / wavenumber)

            q_out[ii, ii] = (self.matrix_A * q_in[ii, ii] + self.matrix_B) * (
                self.matrix_C * q_in[ii, ii] + self.matrix_D
            ) ** (-1)

            Psi_w_out[ii, ii] = 1 / np.conj(q_out[ii, ii]) * wavenumber

        # q_out = q_in
        # Psi_w_out = Psi_w_out2

        return Psi_w_out
