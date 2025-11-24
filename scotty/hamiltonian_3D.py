import logging
import numpy as np
import scotty.derivatives
from scotty.fun_general import (angular_frequency_to_wavenumber,
                                dot,
                                find_normalised_gyro_freq,
                                find_normalised_plasma_freq)
from scotty.geometry_3D import MagneticField_3D_Cartesian
from scotty.logger_3D import arr2str
from scotty.profile_fit import ProfileFitLike
from scotty.typing import ArrayLike, FloatArray
from typing import Dict, Literal, Optional, Tuple

log = logging.getLogger(__name__)

##################################################
#
# HAMILTONIAN CODES
#
##################################################

class DielectricTensor_3D:
    r"""
    Calculates the components of the cold plasma dielectric tensor for a wave with
    angular frequency \Omega:

    \epsilon =
        [ \epilson_{11}   &   -i\epsilon_{12}   &   0            ]
        [ i\epsilon_{12}  &   \epsilon_{11}     &   0            ]
        [ 0               &   0                 &   \epsilon{bb} ]
    
    where
        \epsilon_{11} &= 1 - \frac{\Omega_{pe}^2}{\Omega^2 - \Omega_{ce}^2}
        \epsilon_{12} &= 1 - \frac{\Omega_{pe}^2\Omega_{ce}}{\Omega(\Omega^2 - \Omega_{ce}^2)}
        \epsilon_{bb} &= 1 - \frac{\Omega_{pe}^2}{\Omega^2}
    
    The components of the dielectric tensor are calculated in the
    :math:`(\hat{\mathbf{u}}_1, \hat{\mathbf{u}}_2, \hat{\mathbf{b}})` basis.
    
    Hence, :math:`\epsilon_{11}`, :math:`\epsilon_{12}`, and :math:`\epsilon_{bb}`
    correspond to the ``S``, ``D``, and ``P`` variables in Stix, respectively.
    
    The notation used in this code is chosen to be consistent with Hall-Chen, Parra,
    Hillesheim, PPCF 2022.

    Parameters
    ----------
    electron_density:
        Electron number density
    angular_frequency:
        Angular frequency of the beam
    B_total:
        Magnitude of the magnetic field
    temperature:
        Temperature profile [optional]. Used to calculate relativistic corrections 
        to electron mass, which affects :math:`\Omega_{pe}` and :math: `\Omega_{ce}`.
    """

    def __init__(
        self,
        electron_density: ArrayLike,
        angular_frequency: float,
        B_total: ArrayLike,
        temperature: Optional[ArrayLike] = None):
        
        _plasma_freq_2 = find_normalised_plasma_freq(electron_density, angular_frequency, temperature)**2
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



class Hamiltonian_3D:
    r"""
    Functor to evaluate derivatives of the Hamiltonian, H, at a given set of points.

    Scotty calculates derivatives using a grid-free finite difference approach. The
    Hamiltonian is evaluated at, essentially, an arbitrary set of points around the location
    we wish to get the derivatives at. In practice we define stencils as relative offsets
    from a central point, and the evaluation points are the product of the spacing in a
    given direction with the stencil offsets. By carefully choosing our stencils and
    evaluating all of the derivatives at once, we can reuse evaluations of :math:`H` between
    derivatives, saving a lot of computation.

    The stencils are defined as a `dict` with a `tuple` of offsets as keys and `float`
    weights as values. For example, the `CFD1_stencil`::
        {(1,): 0.5, (-1,): -0.5}

    defines the second-order first central-difference:
        f' = \frac{f(x + \delta_x) - f(x - \delta_x)}{2\delta_x}

    The keys are tuples so that we can iterate over the offsets for the mixed
    second derivatives.

    The stencils have been chosen to maximise the reuse of Hamiltonian
    evaluations without sacrificing accuracy.

    Parameters
    ----------
    launch_angular_frequency
        Angular frequency of the beam
    mode_flag
        Either ``+/-1``, used to determine which mode branch to use
    delta_X
        Finite difference spacing in the ``X`` direction
    delta_Y
        Finite difference spacing in the ``Y`` direction
    delta_Z
        Finite difference spacing in the ``Z`` direction
    delta_K_X
        Finite difference spacing in the ``K_X`` direction
    delta_K_Y
        Finite difference spacing in the ``K_Y`` direction
    delta_K_Z
        Finite difference spacing in the ``K_Z`` direction
    field
        An object describing the magnetic field of the plasma
    density_fit
        Function or ``Callable`` parameterising the density
    temperature_fit
        Function or ``Callable`` parameterising the density. Default is None
    """

    def __init__(self,
                 launch_angular_frequency: float,
                 mode_flag: Literal[1, -1],
                 delta_X: float,
                 delta_Y: float,
                 delta_Z: float,
                 delta_K_X: float,
                 delta_K_Y: float,
                 delta_K_Z: float,
                 field: MagneticField_3D_Cartesian,
                 density_fit: ProfileFitLike,
                 temperature_fit: Optional[ProfileFitLike] = None):
        
        self.angular_frequency = launch_angular_frequency
        self.wavenumber_K0 = angular_frequency_to_wavenumber(launch_angular_frequency)
        self.mode_flag = mode_flag
        self.spacings = {"X": delta_X, "Y": delta_Y, "Z": delta_Z, "K_X": delta_K_X, "K_Y": delta_K_Y, "K_Z": delta_K_Z}
        self.field = field
        self.density = density_fit
        self.temperature = temperature_fit

        log.debug(f"""
        ##################################################
        #
        # Creating Hamiltonian with:
        #   - mode_flag = {self.mode_flag}
        #   - w_launch = {self.angular_frequency}
        #   - K_launch = {self.wavenumber_K0}
        #   - (finite difference) spacings = {self.spacings}
        #   - field type = {type(field)}
        #   - density fit type = {type(density_fit)}
        #   - temperature fit type = {type(temperature_fit)}
        #
        ##################################################
        """)
    
    def __call__(self,
                 X: ArrayLike,
                 Y: ArrayLike,
                 Z: ArrayLike,
                 K_X: ArrayLike,
                 K_Y: ArrayLike,
                 K_Z: ArrayLike):
        
        K_magnitude = np.sqrt(K_X**2 + K_Y**2 + K_Z**2)
        polflux = self.field.polflux(X,Y,Z)
        electron_density = self.density(polflux)

        if self.temperature: temperature = self.temperature(polflux)
        else:                temperature = None

        B_X = np.squeeze(self.field.B_X(X,Y,Z))
        B_Y = np.squeeze(self.field.B_Y(X,Y,Z))
        B_Z = np.squeeze(self.field.B_Z(X,Y,Z))
        B_magnitude = np.sqrt(B_X**2 + B_Y**2 + B_Z**2)
        b_hat = np.array([B_X, B_Y, B_Z]) / B_magnitude
        K_hat = np.array([K_X, K_Y, K_Z]) / K_magnitude

        # Square of the mismatch angle
        if np.size(X) == 1: sin_theta_m_sq = np.dot(b_hat, K_hat)**2
        else:               sin_theta_m_sq = dot(b_hat.T, K_hat.T)**2
        
        epsilon = DielectricTensor_3D(electron_density, self.angular_frequency, B_magnitude, temperature)

        Booker_alpha = (epsilon.e_bb * sin_theta_m_sq) + epsilon.e_11 * (1 - sin_theta_m_sq)
        Booker_beta  = (-epsilon.e_11 * epsilon.e_bb * (1 + sin_theta_m_sq)) - (epsilon.e_11**2 - epsilon.e_12**2) * (1 - sin_theta_m_sq)
        Booker_gamma = epsilon.e_bb * (epsilon.e_11**2 - epsilon.e_12**2)

        H_discriminant = np.maximum(np.zeros_like(Booker_beta), (Booker_beta**2 - 4 * Booker_alpha * Booker_gamma))

        H_Booker = (K_magnitude / self.wavenumber_K0)**2 + (Booker_beta - self.mode_flag * np.sqrt(H_discriminant)) / (2 * Booker_alpha)

        log.trace(f"""
        ##################################################
        #
        # Calling Hamiltonian with:
        #   - [X, Y, Z] = [{X, Y, Z}]
        #   - [K_X, K_Y, K_Z] = [{K_X, K_Y, K_Z}]
        #
        # Calculated values:
        #   - pol. flux = {polflux}
        #   - n_e = {electron_density}
        #   - T_e = {temperature}
        #   - [B_X, B_Y, B_Z] = [{B_X, B_Y, B_Z}]
        #
        #   - sin(theta_m)^2 = {sin_theta_m_sq}
        #   - abs(theta_m) (in rad) = {np.arcsin(np.sqrt(sin_theta_m_sq))}
        #   - abs(theta_m) (in deg) = {np.rad2deg(np.arcsin(np.sqrt(sin_theta_m_sq)))}
        #
        #   - e_11 = {epsilon.e_11}
        #   - e_12 = {epsilon.e_12}
        #   - e_bb = {epsilon.e_bb}
        #
        #   - Booker_a (a) = {Booker_alpha}
        #   - Booker_b (b) = {Booker_beta}
        #   - Booker_g (g) = {Booker_gamma}
        #   - b^2 - 4ag = {H_discriminant}
        #   - H_Booker = {H_Booker}
        #
        ##################################################
        """)
        
        return H_Booker
    
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
        
        if log.isEnabledFor(5):
            _printmsg = "\n".join(f"            #   - {key} = {derivatives[key]}" for key in derivatives)
            log.trace(f"""
            ##################################################
            #
            # Calling Hamiltonian.derivatives with:
            #   - [X, Y, Z] = [{X, Y, Z}]
            #   - [K_X, K_Y, K_Z] = [{K_X, K_Y, K_Z}]
            #
            # Calculated values:
            {_printmsg}
            #
            ##################################################
            """)
        
        return derivatives



def initialise_hamiltonians(launch_angular_frequency: float,
                            delta_X: float,
                            delta_Y: float,
                            delta_Z: float,
                            delta_K_X: float,
                            delta_K_Y: float,
                            delta_K_Z: float,
                            field: MagneticField_3D_Cartesian,
                            density_fit: ProfileFitLike,
                            temperature_fit: Optional[ProfileFitLike] = None):
    
    log.info(f"Initialising the Hamiltonians with `mode_flag`s = +1 and -1")
    
    hamiltonian_pos1 = Hamiltonian_3D(launch_angular_frequency,
                                      1, # mode_flag
                                      delta_X,
                                      delta_Y,
                                      delta_Z,
                                      delta_K_X,
                                      delta_K_Y,
                                      delta_K_Z,
                                      field,
                                      density_fit,
                                      temperature_fit)
    
    hamiltonian_neg1 = Hamiltonian_3D(launch_angular_frequency,
                                      -1, # mode_flag
                                      delta_X,
                                      delta_Y,
                                      delta_Z,
                                      delta_K_X,
                                      delta_K_Y,
                                      delta_K_Z,
                                      field,
                                      density_fit,
                                      temperature_fit)
    
    return hamiltonian_pos1, hamiltonian_neg1



def assign_hamiltonians(mode_flag_launch: Literal["O", "X", 1, -1],
                        mode_flag_initial: Literal[1, -1],
                        hamiltonian_pos1: Hamiltonian_3D,
                        hamiltonian_neg1: Hamiltonian_3D,
                        q_initial_cartesian: FloatArray,
                        K_initial_cartesian: FloatArray,
                        tol_H: float = 1e-5):
    
    log.info(f"Assigning the correct Hamiltonian corresponding to `mode_flag` = {mode_flag_launch}")
    
    H_pos1 = hamiltonian_pos1(*q_initial_cartesian, *K_initial_cartesian)
    H_neg1 = hamiltonian_neg1(*q_initial_cartesian, *K_initial_cartesian)

    log.debug(f"""
        ##################################################
        #
        # For [X,Y,Z] = {arr2str(q_initial_cartesian)} and K = {arr2str(K_initial_cartesian)}:
        #   - H (mode_flag +1) = {H_pos1}
        #   - H (mode_flag -1) = {H_neg1}
        #   - H tolerance      = {tol_H}
        #
        ##################################################
        """)

    # Some second checks make sure everything is in order
    if abs(H_pos1) < tol_H and abs(H_neg1) > tol_H:
        H, H_other, new_mode_flag_initial = hamiltonian_pos1, hamiltonian_neg1, 1
    elif abs(H_pos1) > tol_H and abs(H_neg1) < tol_H:
        H, H_other, new_mode_flag_initial = hamiltonian_neg1, hamiltonian_pos1, -1
    elif abs(H_pos1) > tol_H and abs(H_neg1) > tol_H:
        log.warning(f"Neither `mode_flag` = +1 nor -1 are solutions! (|H| > {tol_H} for both)")
        new_mode_flag_initial = None
    else:
        log.warning(f"Both `mode_flag` = +1 or -1 are solutions! (|H| < {tol_H} for both)")
        new_mode_flag_initial = None
    
    # If `new_mode_flag_initial` is not False-y (i.e. no solution assigned)
    # and the user-specified `mode_flag` (`mode_flag_launch`) is "O" or "X"
    # then this means we are do not assign any Hamiltonian and just raise an
    # error. Otherwise `mode_flag_launch` is +1 or -1, then warn the user but
    # just use that to initialise the corresponding Hamiltonian.
    if not new_mode_flag_initial and mode_flag_launch in ["O", "X"]:
        raise ValueError(f"Unable to determine which `mode_flag` (+1 or -1) to use for {mode_flag_launch}-mode")
    elif mode_flag_initial != new_mode_flag_initial:
        log.warning(f"`mode_flag` was not selected correctly in `find_plasma_entry_parameters` and has been changed from {mode_flag_initial} to {new_mode_flag_initial}. This may or may not be a significant problem.")
        mode_flag_initial = new_mode_flag_initial
    
    log.debug(f"Hamiltonian with computed `mode_flag` = {mode_flag_initial} selected for user-passed `mode_flag` = {mode_flag_launch}")
    
    return H, H_other, mode_flag_initial



def hessians_3D(dH: dict):
    """
    Given a dictionary containing the second derivatives of the Hamiltonian (from
    hamiltonian.derivatives with second_order = True), compute the elements of the
    Hessian of the Hamiltonian:

    Hessian =
        \nabla   \nabla   H
        \nabla_K \nabla   H
        \nabla_K \nabla_K H
    """

    d2H_dX2     = dH["d2H_dX2"]
    d2H_dY2     = dH["d2H_dY2"]
    d2H_dZ2     = dH["d2H_dZ2"]
    d2H_dX_dY   = dH["d2H_dX_dY"]
    d2H_dX_dZ   = dH["d2H_dX_dZ"]
    d2H_dY_dZ   = dH["d2H_dY_dZ"]

    d2H_dKx2    = dH["d2H_dKx2"]
    d2H_dKy2    = dH["d2H_dKy2"]
    d2H_dKz2    = dH["d2H_dKz2"]
    d2H_dKx_dKy = dH["d2H_dKx_dKy"]
    d2H_dKx_dKz = dH["d2H_dKx_dKz"]
    d2H_dKy_dKz = dH["d2H_dKy_dKz"]

    d2H_dX_dKx  = dH["d2H_dX_dKx"]
    d2H_dX_dKy  = dH["d2H_dX_dKy"]
    d2H_dX_dKz  = dH["d2H_dX_dKz"]
    d2H_dY_dKx  = dH["d2H_dY_dKx"]
    d2H_dY_dKy  = dH["d2H_dY_dKy"]
    d2H_dY_dKz  = dH["d2H_dY_dKz"]
    d2H_dZ_dKx  = dH["d2H_dZ_dKx"]
    d2H_dZ_dKy  = dH["d2H_dZ_dKy"]
    d2H_dZ_dKz  = dH["d2H_dZ_dKz"]

    def reshape(array: FloatArray):
        """Such that shape is [points,3,3] instead of [3,3,points]"""
        if array.ndim == 2: return array
        return np.moveaxis(np.squeeze(array), 2, 0)
    
    grad_grad_H = reshape(np.array([
        [d2H_dX2,       d2H_dX_dY,     d2H_dX_dZ],
        [d2H_dX_dY,     d2H_dY2,       d2H_dY_dZ],
        [d2H_dX_dZ,     d2H_dY_dZ,     d2H_dZ2  ]
    ]))

    gradK_grad_H = reshape(np.array([
        [d2H_dX_dKx,    d2H_dY_dKx,    d2H_dZ_dKx],
        [d2H_dX_dKy,    d2H_dY_dKy,    d2H_dZ_dKy],
        [d2H_dX_dKz,    d2H_dY_dKz,    d2H_dZ_dKz]
    ]))

    gradK_gradK_H = reshape(np.array([
        [d2H_dKx2,      d2H_dKx_dKy,   d2H_dKx_dKz],
        [d2H_dKx_dKy,   d2H_dKy2,      d2H_dKy_dKz],
        [d2H_dKx_dKz,   d2H_dKy_dKz,   d2H_dKz2   ]
    ]))

    return grad_grad_H, gradK_grad_H, gradK_gradK_H