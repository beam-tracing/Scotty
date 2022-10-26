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


def make_my_lens(name, lens_type="thin", focal_length=None):

    if lens_type == "thin":
        myLens = Thin_Lens(name)

        if name == "MAST_V_band":
            myLens.focal_length = 0.125
        elif name == "MAST_Q_band":
            myLens.focal_length = 0.27
        elif name == "DBS_UCLA_DIII-D_240":
            myLens.focal_length = focal_length
        elif name == "DBS_UCLA_MAST-U":
            myLens.focal_length = (
                0.1477692307692308  # Harmonic mean of front and back focal distances
            )

    elif lens_type == "thick":
        myLens = Thick_Lens(name)

        if name == "MAST_V_band":
            myLens.focal_length = 0.125
            myLens.thickness = 0.0633
            myLens.ref_index = 1.53
        elif name == "MAST_Q_band":
            myLens.focal_length = 0.27
            myLens.thickness = 0.0338
            myLens.ref_index = 1.53

    elif lens_type == "hyperbolic":
        myLens = ABCD_Lens(name)

        if name == "MAST_V_band":
            myLens.matrix_A = 1.000000000000
            myLens.matrix_B = 41.347131119605 * 10 ** (-3)  # convert
            myLens.matrix_C = -0.006016900790 * 10 ** (3)
            myLens.matrix_D = 0.751218414102
        elif name == "MAST_Q_band":
            myLens.matrix_A = 1.000000000000
            myLens.matrix_B = 22.103343171495 * 10 ** (-3)  # convert
            myLens.matrix_C = -0.003423517875 * 10 ** (3)
            myLens.matrix_D = 0.924328809548

            ## Thick lens
            # myLens.matrix_B = 0.022091503267973853
            # myLens.matrix_C = -3.7037037037037033
            # myLens.matrix_D = 0.9181796175260227

            ## Thin lens
            # myLens.matrix_B = 0.0
            # myLens.matrix_C = -1/0.27
            # myLens.matrix_D = 1.0

    return myLens


class Lens:
    # init method or constructor
    def __init__(self, name):
        self.name = name
        self.focal_length = None

    # Sample Method
    def who_am_I(self):
        print("Lens name: ", self.name)


class Thin_Lens(Lens):
    """
    inherits Lens
    """

    def __init__(self, name):
        # Calling init of parent class
        super().__init__(name)

    def output_beam(self, Psi_w_in, freq_GHz):
        """
        - Psi_w_in should be a 2x2 matrix
        - Assumes the real part of Psi_w is diagonalised
        - Doesn't assume that the diagnoalised Re(Psi_w)'s components are equal
          to each other, that is, the curvatures are allowed to be 'elliptical'
        """
        if self.focal_length is None:
            print("Warning: focal length not initialised")

        angular_frequency = 2 * np.pi * 10.0**9 * freq_GHz
        wavenumber = angular_frequency / constants.c

        Psi_w_out = Psi_w_in - np.eye(2) * wavenumber / self.focal_length

        return Psi_w_out


class Thick_Lens(Lens):
    """
    inherits Lens
    """

    def __init__(self, name):
        # Calling init of parent class
        super().__init__(name)
        self.thickness = None
        self.ref_index = None

    def output_beam(self, Psi_w_in, freq_GHz):
        """
        - Psi_w_in should be a 2x2 matrix
        - Assumes Psi_w is diagonalised
        - Doesn't assume that the diagnoalised Re(Psi_w)'s components are equal
          to each other, that is, the curvatures are allowed to be 'elliptical'
        """
        if self.focal_length is None:
            print("Warning: focal length not initialised")

        angular_frequency = 2 * np.pi * 10.0**9 * freq_GHz
        wavenumber = angular_frequency / constants.c

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

    def __init__(self, name):
        # Calling init of parent class
        super().__init__(name)
        self.matrix_A = None
        self.matrix_B = None
        self.matrix_C = None
        self.matrix_D = None

    def output_beam(self, Psi_w_in, freq_GHz):
        """
        - Psi_w_in should be a 2x2 matrix
        - Assumes Psi_w is diagonalised
        - Doesn't assume that the diagnoalised Re(Psi_w)'s components are equal
          to each other, that is, the curvatures are allowed to be 'elliptical'
        """
        if self.matrix_A is None:
            print("Warning: element A of the ABCD matrix not defined")
        if self.matrix_B is None:
            print("Warning: element A of the ABCD matrix not defined")
        if self.matrix_C is None:
            print("Warning: element A of the ABCD matrix not defined")
        if self.matrix_D is None:
            print("Warning: element A of the ABCD matrix not defined")

        angular_frequency = 2 * np.pi * freq_GHz * 10.0**9
        wavenumber = angular_frequency / constants.c

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
