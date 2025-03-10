import numpy as np
import scotty.derivatives
from scotty.fun_general import (angular_frequency_to_wavenumber,
                                dot,
                                find_normalised_gyro_freq,
                                find_normalised_plasma_freq)
from scotty.profile_fit import ProfileFitLike
from scotty.typing import ArrayLike, FloatArray
from typing import Dict, Optional, Tuple

# new -- for stellarator case
from desc.grid import Grid
from scotty.fun_general_stellarator import xyz2rtz

class DielectricTensor_stellarator:
    def __init__(
        self,
        electron_density: ArrayLike,
        angular_frequency: float,
        B_total: ArrayLike,
        temperature: Optional[ArrayLike] = None):
        
        _plasma_freq_2 = (find_normalised_plasma_freq(electron_density, angular_frequency, temperature)**2)
        _gyro_freq = find_normalised_gyro_freq(B_total, angular_frequency, temperature)
        _gyro_freq_2 = _gyro_freq**2

        self._epsilon_bb = 1 - _plasma_freq_2
        self._epsilon_11 = 1 - _plasma_freq_2 / (1 - _gyro_freq_2)
        self._epsilon_12 = _plasma_freq_2 * _gyro_freq / (1 - _gyro_freq_2)

    @property
    def e_bb(self):
        r"""The :math:`\epsilon_{bb}` component; also called epsilon_para"""
        return self._epsilon_bb

    @property
    def e_11(self):
        r"""The :math:`\epsilon_{11}` component; also called epsilon_perp"""
        return self._epsilon_11

    @property
    def e_12(self):
        r"""The :math:`\epsilon_{12}` component; also called epsilon_g"""
        return self._epsilon_12



class Hamiltonian_stellarator:
    def __init__(self,
                 equilibrium,
                 launch_angular_frequency: float,
                 mode_flag: int,
                 delta_X: float,
                 delta_Y: float,
                 delta_Z: float,
                 delta_K_X: float,
                 delta_K_Y: float,
                 delta_K_Z: float,
                 temperature_fit: Optional[ProfileFitLike] = None):
        
        self.equilibrium = equilibrium
        self.angular_frequency = launch_angular_frequency
        self.wavenumber_K0 = angular_frequency_to_wavenumber(launch_angular_frequency)
        self.mode_flag = mode_flag
        self.temperature = temperature_fit
        self.spacings = {
            "X": delta_X,
            "Y": delta_Y,
            "Z": delta_Z,
            "K_X": delta_K_X,
            "K_Y": delta_K_Y,
            "K_Z": delta_K_Z
        }
    
    def __call__(self,
                 X: ArrayLike,
                 Y: ArrayLike,
                 Z: ArrayLike,
                 K_X: ArrayLike,
                 K_Y: ArrayLike,
                 K_Z: ArrayLike):
        
        # Converting (X,Y,Z) to (rho,theta,zeta) for some point for the DESC equilibrium
        # Store the point in a custom grid because DESC only accepts grids
        rtz_point = np.array(xyz2rtz(self.equilibrium, X, Y, Z))
        rtz_grid = Grid([rtz_point, [0,0,0]])

        # Calculating poloidal flux
        polflux = rtz_point[0]**2

        # Calculating the needed quantities from the DESC equilibrium (B field and pressure)
        data = self.equilibrium.compute(names=("B","p"), grid=rtz_grid)
        B_central,        B        = data["B"]
        pressure_central, pressure = data["p"]

        # Calculating B and K to find sin_sq_theta_m
        B_R, B_T, B_Z = B[0], B[1], B[2]
        B_magnitude = np.sqrt(B_R**2 + B_T**2 + B_Z**2)
        b_hat = np.array([B_R, B_T, B_Z]) / B_magnitude
        K_magnitude = np.sqrt(K_X**2 + K_Y**2 + K_Z**2)
        K_hat = np.array([K_X, K_Y, K_Z]) / K_magnitude
        sin_theta_m_sq = np.dot(b_hat, K_hat)**2

        # Calculating ne as a function of pressure because I dont have the actual ne profile
        # Choice of 2.1 x 10^19 on the magnetic axis is from Figure 11 of
        # https://scipub.euro-fusion.org/wp-content/uploads/eurofusion/WPS1PR17_18033_submitted.pdf
        electron_density = (pressure/pressure_central)*(2.1) # 10^19 m-3
        
        temperature = None
        epsilon = DielectricTensor_stellarator(electron_density, self.angular_frequency, B_magnitude, temperature)

        Booker_alpha = (epsilon.e_bb * sin_theta_m_sq) + epsilon.e_11 * (1 - sin_theta_m_sq)

        Booker_beta = (-epsilon.e_11 * epsilon.e_bb * (1 + sin_theta_m_sq)) - (epsilon.e_11**2 - epsilon.e_12**2) * (1 - sin_theta_m_sq)

        Booker_gamma = epsilon.e_bb * (epsilon.e_11**2 - epsilon.e_12**2)

        H_discriminant = np.maximum(np.zeros_like(Booker_beta), (Booker_beta**2 - 4 * Booker_alpha * Booker_gamma))

        """
        # TO REMOVE in the future -- for debugging only
        
        # # print()
        # print("X, Y, Z: ", X, Y, Z)
        # print("Kx, Ky, Kz: ", K_X, K_Y, K_Z)
        # # print("K: ", K_magnitude)
        # print("K_hat: ", K_hat)
        # print("Bx, By, Bz: ", B_X, B_Y, B_Z)
        # # print("B: ", B_magnitude)
        # print("b_hat: ", b_hat)
        # # print("polflux: ", polflux)
        # # print("electron density: ", electron_density)
        # # print("epsilon e11, e12, ebb: ", epsilon.e_11, epsilon.e_12, epsilon.e_bb)
        # # print("alpha, beta, gamma: ", Booker_alpha, Booker_beta, Booker_gamma)
        # # print("H_discriminant: ", H_discriminant)
        # print("H: ", (K_magnitude / self.wavenumber_K0) ** 2 + (Booker_beta - self.mode_flag * np.sqrt(H_discriminant)) / (2 * Booker_alpha))
        # # print("sin_theta_m_sq", sin_theta_m_sq)
        # print("theta_m", np.arcsin(np.sqrt(sin_theta_m_sq)))
        # # print()
        """

        return (K_magnitude / self.wavenumber_K0) ** 2 + (Booker_beta - self.mode_flag * np.sqrt(H_discriminant)) / (2 * Booker_alpha)
    
    def derivatives(self,
                    X: ArrayLike,
                    Y: ArrayLike,
                    Z: ArrayLike,
                    K_X: ArrayLike,
                    K_Y: ArrayLike,
                    K_Z: ArrayLike,
                    second_order: bool = False) -> Dict[str, ArrayLike]:
        
        """
        Evaluate the first-order derivative in all directions at the given
        point(s), and optionally the second-order ones too
        """

        starts = {"X": X, "Y": Y, "Z": Z, "K_X": K_X, "K_Y": K_Y, "K_Z": K_Z}

        def apply_stencil(dims: Tuple[str, ...], stencil: str):
            return scotty.derivatives.derivative(self, dims, starts, self.spacings, stencil)

        derivatives = {
            "dH_dX":  apply_stencil(("X"), "d1_FFD2"),
            "dH_dY":  apply_stencil(("Y"), "d1_FFD2"),
            "dH_dZ":  apply_stencil(("Z"), "d1_FFD2"),
            "dH_dKx": apply_stencil(("K_X"), "d1_CFD2"),
            "dH_dKy": apply_stencil(("K_Y"), "d1_CFD2"),
            "dH_dKz": apply_stencil(("K_Z"), "d1_CFD2"),
        }

        if second_order:
            second_derivatives = {
                "d2H_dX2":     apply_stencil(("X","X"), "d2_FFD2"),
                "d2H_dY2":     apply_stencil(("Y","Y"), "d2_FFD2"),
                "d2H_dZ2":     apply_stencil(("Z","Z"), "d2_FFD2"),
                "d2H_dX_dY":   apply_stencil(("X","Y"), "d1d1_FFD_FFD2"),
                "d2H_dX_dZ":   apply_stencil(("X","Z"), "d1d1_FFD_FFD2"),
                "d2H_dY_dZ":   apply_stencil(("Y","Z"), "d1d1_FFD_FFD2"),

                "d2H_dKx2":    apply_stencil(("K_X","K_X"), "d2_CFD2"),
                "d2H_dKy2":    apply_stencil(("K_Y","K_Y"), "d2_CFD2"),
                "d2H_dKz2":    apply_stencil(("K_Z","K_Z"), "d2_CFD2"), 
                "d2H_dKx_dKy": apply_stencil(("K_X","K_Y"), "d1d1_CFD_CFD2"),
                "d2H_dKx_dKz": apply_stencil(("K_X","K_Z"), "d1d1_CFD_CFD2"),
                "d2H_dKy_dKz": apply_stencil(("K_Y","K_Z"), "d1d1_CFD_CFD2"),

                "d2H_dX_dKx":  apply_stencil(("X","K_X"), "d1d1_FFD_CFD2"),
                "d2H_dX_dKy":  apply_stencil(("X","K_Y"), "d1d1_FFD_CFD2"),
                "d2H_dX_dKz":  apply_stencil(("X","K_Z"), "d1d1_FFD_CFD2"),
                "d2H_dY_dKx":  apply_stencil(("Y","K_X"), "d1d1_FFD_CFD2"),
                "d2H_dY_dKy":  apply_stencil(("Y","K_Y"), "d1d1_FFD_CFD2"),
                "d2H_dY_dKz":  apply_stencil(("Y","K_Z"), "d1d1_FFD_CFD2"),
                "d2H_dZ_dKx":  apply_stencil(("Z","K_X"), "d1d1_FFD_CFD2"),
                "d2H_dZ_dKy":  apply_stencil(("Z","K_Y"), "d1d1_FFD_CFD2"),
                "d2H_dZ_dKz":  apply_stencil(("Z","K_Z"), "d1d1_FFD_CFD2"),
            }
            derivatives.update(second_derivatives)
        
        """
        # TO REMOVE -- for debugging only
        print(np.array((X,Y,Z)))
        print(np.array((K_X,K_Y,K_Z)))
        for key, value in derivatives.items():
            print(key, value)
        """
        
        return derivatives