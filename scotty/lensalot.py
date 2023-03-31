# -*- coding: utf-8 -*-
# Copyright 2021 - 2023, Valerian Hall-Chen and the Scotty contributors
# SPDX-License-Identifier: GPL-3.0

r"""
Transfer matrix (2x2):

.. math::
    \begin{pmatrix}
        r' \\
        \theta' \\
    \end{pmatrix} =
    \begin{pmatrix}
        A & B \\
        C & D \\
    \end{pmatrix}
    \begin{pmatrix}
        r \\
        \theta \\
    \end{pmatrix}

Consider a ray entering and leaving an optical system. It is distance
:math:`r` above the system axis, entering at an angle :math:`\theta`. It leaves
at a distance :math:`r'`, and an angle :math:`\theta'`
"""

from .fun_general import freq_GHz_to_wavenumber

import numpy as np
from typing import Optional, Dict


def _check_matrix_is_2x2(Psi_w_in):
    """Raise an exception if input matrix is not 2x2"""
    if Psi_w_in.shape != (2, 2):
        raise ValueError(
            f"`Psi_w_in` must be 2x2 matrix, actual shape: {Psi_w_in.shape}"
        )


class Lens:
    """Abstract base class for different lens approximations"""

    def __init__(self, name: str, focal_length: float):
        self.name = name
        self.focal_length = focal_length

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
    """Thin lens approximation"""

    def __init__(self, name, focal_length):
        super().__init__(name, focal_length)

    def output_beam(self, Psi_w_in, freq_GHz):
        _check_matrix_is_2x2(Psi_w_in)
        wavenumber = freq_GHz_to_wavenumber(freq_GHz)
        Psi_w_out = Psi_w_in - np.eye(2) * wavenumber / self.focal_length
        return Psi_w_out

    def __repr__(self):
        return f"Thin_lens('{self.name}', focal_length={self.focal_length})"


class Thick_Lens(Lens):
    """Lens with finite thickness and refractive index"""

    def __init__(self, name, focal_length, thickness, ref_index):
        super().__init__(name, focal_length)
        self.thickness = thickness
        self.ref_index = ref_index

    def output_beam(self, Psi_w_in, freq_GHz):
        _check_matrix_is_2x2(Psi_w_in)
        wavenumber = freq_GHz_to_wavenumber(freq_GHz)

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

    def __repr__(self):
        return (
            f"Thick_Lens('{self.name}', focal_length={self.focal_length}, "
            f"thickness={self.thickness}, ref_index={self.ref_index})"
        )


class ABCD_Lens(Lens):
    """Generalised lens"""

    def __init__(self, name, A, B, C, D):
        self.name = name
        self.matrix_A = A
        self.matrix_B = B
        self.matrix_C = C
        self.matrix_D = D

    def output_beam(self, Psi_w_in, freq_GHz):
        _check_matrix_is_2x2(Psi_w_in)
        wavenumber = freq_GHz_to_wavenumber(freq_GHz)

        q_out = np.zeros_like(Psi_w_in, dtype="complex128")
        q_in = np.zeros_like(Psi_w_in, dtype="complex128")
        Psi_w_out = np.zeros_like(Psi_w_in, dtype="complex128")

        for ii in range(0, 2):
            q_in[ii, ii] = 1 / (np.conj(Psi_w_in[0, 0]) / wavenumber)

            q_out[ii, ii] = (self.matrix_A * q_in[ii, ii] + self.matrix_B) * (
                self.matrix_C * q_in[ii, ii] + self.matrix_D
            ) ** (-1)

            Psi_w_out[ii, ii] = 1 / np.conj(q_out[ii, ii]) * wavenumber

        return Psi_w_out

    def __repr__(self):
        return (
            f"ABCD_Lens('{self.name}', A={self.A}, B={self.B}, C={self.C}, D={self.D}"
        )


def make_my_lens(
    name: str, lens_type: str = "thin", focal_length: Optional[float] = None
) -> Lens:
    """Create one of a pre-existing set of parameterised lenses.

    Arguments
    =========
    name:
        Name of the known lenses
    lens_type:
        One of "thin", "thick", "hyperbolic"
    focal_length:
        Focal length for the DBS_UCLA_DIII-D_240 lens
    """

    lenses: Dict[str, Dict[str, Lens]] = {
        "thin": {
            "MAST_V_band": Thin_Lens("MAST_V_band", 0.125),
            "MAST_Q_band": Thin_Lens("MAST_Q_band", 0.27),
            "DBS_UCLA_DIII-D_240": Thin_Lens("DBS_UCLA_DIII-D_240", focal_length),
            # Harmonic mean of front and back focal distances
            "DBS_UCLA_MAST-U": Thin_Lens("DBS_UCLA_MAST-U", 0.1477692307692308),
        },
        "thick": {
            "MAST_V_band": Thick_Lens(
                "MAST_V_band", focal_length=0.125, thickness=0.0633, ref_index=1.53
            ),
            "MAST_Q_band": Thick_Lens(
                "MAST_Q_band", focal_length=0.27, thickness=0.0338, ref_index=1.53
            ),
        },
        "hyperbolic": {
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
        },
    }

    try:
        lens_type_dict = lenses[lens_type]
    except KeyError:
        raise ValueError(
            f"Unknown lens type '{lens_type}', expected one of {lenses.keys()}"
        )

    try:
        return lens_type_dict[name]
    except KeyError:
        raise ValueError(
            f"Unknown {lens_type} lens '{name}', expected one of {lens_type_dict.keys()}"
        )
