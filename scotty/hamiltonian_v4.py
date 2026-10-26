import logging
from multiprocessing import Value
import numpy as np
from scotty.checks_v4 import VALID_FIELDS, VALID_LAUNCH_MODE_FLAGS
from scotty.derivatives import derivative
from scotty.fun_general_v4 import find_normalised_plasma_ang_freq, find_normalised_gyro_ang_freq, angular_frequency_to_wavenumber, dot, find_Booker_terms, reshape_33N_to_N33
from scotty.geometry_v4 import MagneticField_Cylindrical, MagneticField_Cartesian
from scotty.profile_fit import ProfileFitLike
from scotty.typing import ArrayLike, FloatArray
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Union, cast

log = logging.getLogger(__name__)

##################################################
#
# DIELECTRIC CLASS
#
##################################################

class DielectricTensor:
    def __init__(
        self,
        launch_angular_freq: float,
        B_magnitude: ArrayLike,
        electron_density: ArrayLike,
        temperature: Optional[ArrayLike] = None):

        plasma_freq_2 = find_normalised_plasma_ang_freq(launch_angular_freq, electron_density, temperature)**2
        gyro_freq = find_normalised_gyro_ang_freq(launch_angular_freq, B_magnitude, temperature)
        gyro_freq_2 = gyro_freq**2

        self._epsilon_bb = 1 - plasma_freq_2
        self._epsilon_11 = 1 - plasma_freq_2 / (1 - gyro_freq_2)
        self._epsilon_12 = plasma_freq_2 * gyro_freq / (1 - gyro_freq_2)
    
    @property
    def e_bb(self) -> ArrayLike: return self._epsilon_bb

    @property
    def e_11(self) -> ArrayLike: return self._epsilon_11

    @property
    def e_12(self) -> ArrayLike: return self._epsilon_12

##################################################
#
# HAMILTONIAN CODES
#
##################################################

class Hamiltonian:
    r"""Functor to evaluate derivatives of the Hamiltonian, H, at a given set
    of points.

    Scotty calculates derivatives using a grid-free finite difference approach. The
    Hamiltonian is evaluated at, essentially, an arbitrary set of points around the
    location we wish to get the derivatives at. In practice we define stencils as
    relative offsets from a central point, and the evaluation points are the product
    of the spacing in a given direction with the stencil offsets. By carefully
    choosing our stencils and evaluating all of the derivatives at once, we can
    reuse evaluations of :math:`H` between derivatives, saving a lot of computation.

    The stencils are defined as a `dict` with a `tuple` of offsets as keys and `float`
    weights as values. For example, the `CFD1_stencil`::
        {(1,): 0.5, (-1,): -0.5}

    defines the second-order first central-difference:
        f' = \frac{f(x + \delta_x) - f(x - \delta_x)}{2\delta_x}

    The keys are tuples so that we can iterate over the offsets for the mixed
    second derivatives.

    The stencils have been chosen to maximise the reuse of Hamiltonian
    evaluations without sacrificing accuracy.
    """

    wavenumber: float

    def __init__(
        self,
        launch_angular_freq: float,
        mode_flag: Literal[1, -1],
        deltas: FloatArray,
        field: VALID_FIELDS,
        density_fit: ProfileFitLike,
        temperature_fit: Optional[ProfileFitLike] = None):
    
        self.angular_frequency = launch_angular_freq
        self.wavenumber = angular_frequency_to_wavenumber(launch_angular_freq)
        self.mode_flag = mode_flag
        self.field = field
        self.density = density_fit
        self.temperature = temperature_fit

        if isinstance(field, MagneticField_Cylindrical):
            delta_R, delta_Z, delta_K_R, delta_K_zeta, delta_K_Z = deltas
            self.spacings = {"q0": delta_R, "q1": delta_R, "q2": delta_Z, "K0": delta_K_R, "K1": delta_K_zeta, "K2": delta_K_Z}
            self.delta_R = delta_R
            self.delta_Z = delta_Z
            self.delta_K_R = delta_K_R
            self.delta_K_zeta = delta_K_zeta
            self.delta_K_Z = delta_K_Z
            def _K_vec(K_R: ArrayLike, K_zeta: ArrayLike, K_Z: ArrayLike, q_R: ArrayLike) -> ArrayLike: return np.array([K_R, K_zeta/q_R, K_Z]) # type: ignore

        elif isinstance(field, MagneticField_Cartesian):
            delta_X, delta_Y, delta_Z, delta_K_X, delta_K_Y, delta_K_Z = deltas
            self.spacings = {"q0": delta_X, "q1": delta_Y, "q2": delta_Z, "K0": delta_K_X, "K1": delta_K_Y, "K2": delta_K_Z}
            self.delta_X = delta_X
            self.delta_Y = delta_Y
            self.delta_Z = delta_Z
            self.delta_K_X = delta_K_X
            self.delta_K_Y = delta_K_Y
            self.delta_K_Z = delta_K_Z
            def _K_vec(K_X: ArrayLike, K_Y: ArrayLike, K_Z: ArrayLike, q_R: ArrayLike) -> ArrayLike: return np.array([K_X, K_Y, K_Z]) # type: ignore
        
        self._K_vec = _K_vec
        
        log.debug(f"""
        ##################################################
        #
        # Creating Hamiltonian with:
        #   - mode_flag = {self.mode_flag}
        #   - |w_launch| = {self.angular_frequency}
        #   - |K_launch| = {self.wavenumber}
        #   - (finite difference) spacings = {self.spacings}
        #   - field type = {type(field)}
        #   - density fit type = {type(density_fit)}
        #   - temperature fit type = {type(temperature_fit)}
        #
        ##################################################
        """)

    def __call__(self, q0: FloatArray, q1: FloatArray, q2: Optional[FloatArray] = None, K0: Optional[FloatArray] = None, K1: Optional[FloatArray] = None, K2: Optional[FloatArray] = None) -> FloatArray:

        # This is required because we do abstraction, but derivatives doesnt like that (it requires the names of the arguments)
        if   all(a is not None for a in [q0, q1]) and all(a is None for a in [q2, K0, K1, K2]): (q0, q1, q2), (K0, K1, K2) = q0, q1
        elif all(a is not None for a in [q0, q1, q2, K0, K1, K2]): pass # do nothing
        else: raise RuntimeError(f"Expected either 2 or 6 arguments but got {sum(a is None for a in [q0, q1, q2, K0, K1, K2])}")

        polflux = self.field.polflux(q0, q1, q2)
        electron_density = self.density(polflux)
        temperature = self.temperature(polflux) if self.temperature else None

        B_magnitude = self.field.magnitude(q0, q1, q2) # type: ignore
        b_hat = self.field.unitvector(q0, q1, q2) # type: ignore
        K_vec = self._K_vec(K0, K1, K2, q_R=q0) # type: ignore
        K_magnitude = np.linalg.norm(K_vec, axis=0)
        K_hat = K_vec / K_magnitude

        sin_theta_m = np.dot(b_hat, K_hat) if np.size(q0) == 1 else dot(b_hat, K_hat.T) # when given multiple points, b_hat.shape is (len_tau, N) while K_hat.shape (N, len_tau)
        sin_theta_m_sq = cast(ArrayLike, sin_theta_m**2)

        Booker_alpha, Booker_beta, Booker_gamma = find_Booker_terms(
            launch_angular_freq = self.angular_frequency,
            B_total = B_magnitude,
            sin_theta_m_sq = sin_theta_m_sq,
            electron_density = electron_density,
            temperature = temperature,
        )

        H_discriminant = np.maximum(np.zeros_like(Booker_beta), Booker_beta**2 - 4 * Booker_alpha * Booker_gamma)

        H_Booker = (K_magnitude / self.wavenumber)**2 + (Booker_beta - self.mode_flag * np.sqrt(H_discriminant)) / (2 * Booker_alpha)

        log.trace(f"""
            ##################################################
            #
            # Calling Hamiltonian with:
            #   - {"[R, zeta, Z]" if isinstance(self.field, MagneticField_Cylindrical) else "[X, Y, Z]"} = {q0, q1, q2}
            #   - {"[K_R, K_zeta, K_Z]" if isinstance(self.field, MagneticField_Cylindrical) else "[K_X, K_Y, K_Z]"} = {K0, K1, K2}
            #
            # Calculated values:
            #   - pol. flux = {polflux}
            #   - n_e = {electron_density}
            #   - T_e = {temperature}
            #   - {"[B_R, B_T, B_Z]" if isinstance(self.field, MagneticField_Cylindrical) else "[B_X, B_Y, B_Z]"} = {B_magnitude*b_hat.T}
            #
            #   - sin(theta_m)^2 = {sin_theta_m_sq}
            #   - |theta_m| (in rad) = {np.arcsin(np.sqrt(sin_theta_m_sq))}
            #   - |theta_m| (in deg) = {np.rad2deg(np.arcsin(np.sqrt(sin_theta_m_sq)))}
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
    
    def derivatives(self, q: FloatArray, K: FloatArray, second_order: bool = False) -> Dict[str, ArrayLike]:
        """
        Evaluate the first-order derivative in all directions at the given
        point(s), and optionally the second-order ones too

        Parameters
        ----------
        q : (3,N) FloatArray
            Position vector, lab frame
        K : (3,N) FloatArray
            Wavevector, lab frame
        second_order : bool
            `False` returns only first order derivatives; `True` returns
            first and second order derivatives

        Returns
        ----------
        dH : dict
            Dictionary whose keys are the derivative names; e.g.
            `dH_dX`, `dH_dKX`, etc.
        """
        
        def apply_stencil(dims: Tuple[str, ...], stencil: str): return derivative(self, dims, starts, self.spacings, stencil)

        # Capture the location we want the derivatives at
        if isinstance(self.field, MagneticField_Cylindrical):
            R, zeta, Z = q
            K_R, K_zeta, K_Z = K
            starts = {"q0": R, "q1": zeta, "q2": Z, "K0": K_R, "K1": K_zeta, "K2": K_Z}

            dH = {
                "dH_dR":     (dH_dR := apply_stencil(("q0",), "d1_FFD2")),
                "dH_dzeta":  np.zeros_like(dH_dR),
                "dH_dZ":     apply_stencil(("q2",), "d1_FFD2"),
                "dH_dKR":    apply_stencil(("K0",), "d1_CFD2"),
                "dH_dKzeta": apply_stencil(("K1",), "d1_CFD2"),
                "dH_dKZ":    apply_stencil(("K2",), "d1_CFD2"),
            }

            if second_order:
                dH.update({
                    "d2H_dR2":        apply_stencil(("q0", "q0"), "d2_FFD2"),
                    "d2H_dZ2":        apply_stencil(("q2", "q2"), "d2_FFD2"),
                    "d2H_dKR2":       apply_stencil(("K0", "K0"), "d2_CFD2"),
                    "d2H_dKzeta2":    apply_stencil(("K1", "K1"), "d2_CFD2"),
                    "d2H_dKZ2":       apply_stencil(("K2", "K2"), "d2_CFD2"),
                    "d2H_dR_dZ":      apply_stencil(("q0", "q2"), "d1d1_FFD_FFD2"),
                    "d2H_dR_dKR":     apply_stencil(("q0", "K0"), "d1d1_FFD_CFD2"),
                    "d2H_dR_dKzeta":  apply_stencil(("q0", "K1"), "d1d1_FFD_CFD2"),
                    "d2H_dR_dKZ":     apply_stencil(("q0", "K2"), "d1d1_FFD_CFD2"),
                    "d2H_dZ_dKR":     apply_stencil(("q2", "K0"), "d1d1_FFD_CFD2"),
                    "d2H_dZ_dKzeta":  apply_stencil(("q2", "K1"), "d1d1_FFD_CFD2"),
                    "d2H_dZ_dKZ":     apply_stencil(("q2", "K2"), "d1d1_FFD_CFD2"),
                    "d2H_dKR_dKZ":    apply_stencil(("K0", "K2"), "d1d1_CFD_CFD2"),
                    "d2H_dKR_dKzeta": apply_stencil(("K0", "K1"), "d1d1_CFD_CFD2"),
                    "d2H_dKzeta_dKZ": apply_stencil(("K1", "K2"), "d1d1_CFD_CFD2"),
                })
        
        # equivalent to elif isinstance(self.field, MagneticField_Cartesian):
        # but written as else to stop the type checker complaining
        else:
            X, Y, Z = q
            K_X, K_Y, K_Z = K
            starts = {"q0": X, "q1": Y, "q2": Z, "K0": K_X, "K1": K_Y, "K2": K_Z}

            dH = {
                "dH_dX":     apply_stencil(("q0",), "d1_FFD2"),
                "dH_dY":     apply_stencil(("q1",), "d1_FFD2"),
                "dH_dZ":     apply_stencil(("q2",), "d1_FFD2"),
                "dH_dKX":    apply_stencil(("K0",), "d1_CFD2"),
                "dH_dKY":    apply_stencil(("K1",), "d1_CFD2"),
                "dH_dKZ":    apply_stencil(("K2",), "d1_CFD2"),
            }

            if second_order:
                dH.update({
                    "d2H_dX2":     apply_stencil(("q0", "q0"), "d2_FFD2"),
                    "d2H_dY2":     apply_stencil(("q1", "q1"), "d2_FFD2"),
                    "d2H_dZ2":     apply_stencil(("q2", "q2"), "d2_FFD2"),
                    "d2H_dX_dY":   apply_stencil(("q0", "q1"), "d1d1_FFD_FFD2"),
                    "d2H_dX_dZ":   apply_stencil(("q0", "q2"), "d1d1_FFD_FFD2"),
                    "d2H_dY_dZ":   apply_stencil(("q1", "q2"), "d1d1_FFD_FFD2"),
                    "d2H_dKX2":    apply_stencil(("K0", "K0"), "d2_CFD2"),
                    "d2H_dKY2":    apply_stencil(("K1", "K1"), "d2_CFD2"),
                    "d2H_dKZ2":    apply_stencil(("K2", "K2"), "d2_CFD2"),
                    "d2H_dKX_dKY": apply_stencil(("K0", "K1"), "d1d1_CFD_CFD2"),
                    "d2H_dKX_dKZ": apply_stencil(("K0", "K2"), "d1d1_CFD_CFD2"),
                    "d2H_dKY_dKZ": apply_stencil(("K1", "K2"), "d1d1_CFD_CFD2"),
                    "d2H_dX_dKX":  apply_stencil(("q0", "K0"), "d1d1_FFD_CFD2"),
                    "d2H_dX_dKY":  apply_stencil(("q0", "K1"), "d1d1_FFD_CFD2"),
                    "d2H_dX_dKZ":  apply_stencil(("q0", "K2"), "d1d1_FFD_CFD2"),
                    "d2H_dY_dKX":  apply_stencil(("q1", "K0"), "d1d1_FFD_CFD2"),
                    "d2H_dY_dKY":  apply_stencil(("q1", "K1"), "d1d1_FFD_CFD2"),
                    "d2H_dY_dKZ":  apply_stencil(("q1", "K2"), "d1d1_FFD_CFD2"),
                    "d2H_dZ_dKX":  apply_stencil(("q2", "K0"), "d1d1_FFD_CFD2"),
                    "d2H_dZ_dKY":  apply_stencil(("q2", "K1"), "d1d1_FFD_CFD2"),
                    "d2H_dZ_dKZ":  apply_stencil(("q2", "K2"), "d1d1_FFD_CFD2"),
                })

        if log.isEnabledFor(5):
            _printmsg = "\n".join(f"            #   - {k} = {v}" for k, v in dH.items())
            log.trace(f"""
            ##################################################
            #
            # Calling Hamiltonian.derivatives with:
            #   - {"[R, zeta, Z]" if isinstance(self.field, MagneticField_Cylindrical) else "[X, Y, Z]"} = {q}
            #   - {"[K_R, K_zeta, K_Z]" if isinstance(self.field, MagneticField_Cylindrical) else "[K_X, K_Y, K_Z]"} = {K}
            #
            # Calculated values: \n{_printmsg}
            #
            ##################################################
            """)
        
        return dH



def initialise_hamiltonians(
    launch_angular_freq: float,
    deltas: FloatArray,
    field: VALID_FIELDS,
    density_fit: ProfileFitLike,
    temperature_fit: Optional[ProfileFitLike] = None,
) -> Tuple[Hamiltonian, Hamiltonian]:
    
    log.info(f"Initialising Hamiltonians for `mode_flag` = +1 and -1")

    H_pos1 = Hamiltonian(
        launch_angular_freq = launch_angular_freq,
        mode_flag = 1,
        deltas = deltas,
        field = field,
        density_fit = density_fit,
        temperature_fit = temperature_fit)

    H_neg1 = Hamiltonian(
        launch_angular_freq = launch_angular_freq,
        mode_flag = -1,
        deltas = deltas,
        field = field,
        density_fit = density_fit,
        temperature_fit = temperature_fit)
    
    return H_pos1, H_neg1



def assign_hamiltonians(
    mode_flag_initial: Literal[1, -1],
    hamiltonian_pos1: Hamiltonian,
    hamiltonian_neg1: Hamiltonian,
    q_initial: FloatArray,
    K_initial: FloatArray,
    tol_H: float = 1e-5,
) -> Tuple[Hamiltonian, Hamiltonian]:
    
    log.debug(f"Assigning the correct Hamiltonian corresponding to `mode_flag_initial` = {mode_flag_initial}")

    if mode_flag_initial ==  1:
        H, H_other = hamiltonian_pos1, hamiltonian_neg1
    else: # mode_flag_initial == -1:
        H, H_other = hamiltonian_neg1, hamiltonian_pos1

    # Checking to make sure H = 0 (i.e. it is indeed the correct solution)
    H_val = H(q_initial, K_initial)
    H_other_val = H_other(q_initial, K_initial)
    if H_val > tol_H and H_val > H_other_val:
        log.warning(f"`mode_flag` and `Hamiltonian` may not be selected correctly: H = {H_val} > H_tol = {tol_H} for `mode_flag_initial` = {mode_flag_initial}")
        log.warning(f"This may or may not cause issues")

    return H, H_other



def hessians(dH: dict, cartesian: bool) -> Tuple[FloatArray, FloatArray, FloatArray]:
    r"""
    Given a dictionary containing the second derivatives of the Hamiltonian (from
    hamiltonian.derivatives with second_order = True), compute the elements of the
    Hessian of the Hamiltonian:

    .. math::
          \begin{gather}
            \nabla \nabla H \\
            \nabla_K \nabla H \\
            \nabla_K \nabla_K H \\
          \end{gather}

    Parameters
    ----------
    dH : dict
        Dictionary of first and second derivatives, generated via the derivative()
        function in the Hamiltonian() class
    cartesian : bool
        `False` returns cylindrical derivatives; `True` returns Cartesian derivatives

    Returns
    ----------
    grad_grad_H : (3,3) or (N,3,3) FloatArray
        Hessian with spatial derivatives
    gradK_grad_H : (3,3) or (N,3,3) FloatArray
        Hessian with mixed derivatives
    gradK_gradK_H : (3,3) or (N,3,3) FloatArray
        Hessian with wavevector derivatives
    """

    if not cartesian: # isinstance(field, MagneticField_Cylindrical):
        d2H_dR2        = dH["d2H_dR2"]
        d2H_dZ2        = dH["d2H_dZ2"]
        d2H_dKR2       = dH["d2H_dKR2"]
        d2H_dKzeta2    = dH["d2H_dKzeta2"]
        d2H_dKZ2       = dH["d2H_dKZ2"]
        d2H_dR_dZ      = dH["d2H_dR_dZ"]
        d2H_dKR_dR     = dH["d2H_dR_dKR"]
        d2H_dKzeta_dR  = dH["d2H_dR_dKzeta"]
        d2H_dKZ_dR     = dH["d2H_dR_dKZ"]
        d2H_dKR_dZ     = dH["d2H_dZ_dKR"]
        d2H_dKzeta_dZ  = dH["d2H_dZ_dKzeta"]
        d2H_dKZ_dZ     = dH["d2H_dZ_dKZ"]
        d2H_dKR_dKZ    = dH["d2H_dKR_dKZ"]
        d2H_dKR_dKzeta = dH["d2H_dKR_dKzeta"]
        d2H_dKzeta_dKZ = dH["d2H_dKzeta_dKZ"]

        zeros = np.zeros_like(d2H_dR2)

        grad_grad_H = reshape_33N_to_N33(np.array([
            [d2H_dR2,        zeros,          d2H_dR_dZ     ],
            [zeros,          zeros,          zeros         ],
            [d2H_dR_dZ,      zeros,          d2H_dZ2       ],
        ]))

        gradK_grad_H = reshape_33N_to_N33(np.array([
            [d2H_dKR_dR,     zeros,          d2H_dKR_dZ    ],
            [d2H_dKzeta_dR,  zeros,          d2H_dKzeta_dZ ],
            [d2H_dKZ_dR,     zeros,          d2H_dKZ_dZ    ],
        ]))

        gradK_gradK_H = reshape_33N_to_N33(np.array([
            [d2H_dKR2,       d2H_dKR_dKzeta, d2H_dKR_dKZ   ],
            [d2H_dKR_dKzeta, d2H_dKzeta2,    d2H_dKzeta_dKZ],
            [d2H_dKR_dKZ,    d2H_dKzeta_dKZ, d2H_dKZ2      ],
        ]))
    
    # equivalent to elif isinstance(self.field, MagneticField_Cartesian):
    # but written as else to stop the type checker complaining
    else:
        d2H_dX2     = dH["d2H_dX2"]
        d2H_dY2     = dH["d2H_dY2"]
        d2H_dZ2     = dH["d2H_dZ2"]
        d2H_dX_dY   = dH["d2H_dX_dY"]
        d2H_dX_dZ   = dH["d2H_dX_dZ"]
        d2H_dY_dZ   = dH["d2H_dY_dZ"]
        d2H_dKX2    = dH["d2H_dKX2"]
        d2H_dKY2    = dH["d2H_dKY2"]
        d2H_dKZ2    = dH["d2H_dKZ2"]
        d2H_dKX_dKY = dH["d2H_dKX_dKY"]
        d2H_dKX_dKZ = dH["d2H_dKX_dKZ"]
        d2H_dKY_dKZ = dH["d2H_dKY_dKZ"]
        d2H_dX_dKX  = dH["d2H_dX_dKX"]
        d2H_dX_dKY  = dH["d2H_dX_dKY"]
        d2H_dX_dKZ  = dH["d2H_dX_dKZ"]
        d2H_dY_dKX  = dH["d2H_dY_dKX"]
        d2H_dY_dKY  = dH["d2H_dY_dKY"]
        d2H_dY_dKZ  = dH["d2H_dY_dKZ"]
        d2H_dZ_dKX  = dH["d2H_dZ_dKX"]
        d2H_dZ_dKY  = dH["d2H_dZ_dKY"]
        d2H_dZ_dKZ  = dH["d2H_dZ_dKZ"]
        
        grad_grad_H = reshape_33N_to_N33(np.array([
            [d2H_dX2,       d2H_dX_dY,     d2H_dX_dZ],
            [d2H_dX_dY,     d2H_dY2,       d2H_dY_dZ],
            [d2H_dX_dZ,     d2H_dY_dZ,     d2H_dZ2  ]
        ]))

        gradK_grad_H = reshape_33N_to_N33(np.array([
            [d2H_dX_dKX,    d2H_dY_dKX,    d2H_dZ_dKX],
            [d2H_dX_dKY,    d2H_dY_dKY,    d2H_dZ_dKY],
            [d2H_dX_dKZ,    d2H_dY_dKZ,    d2H_dZ_dKZ]
        ]))

        gradK_gradK_H = reshape_33N_to_N33(np.array([
            [d2H_dKX2,      d2H_dKX_dKY,   d2H_dKX_dKZ],
            [d2H_dKX_dKY,   d2H_dKY2,      d2H_dKY_dKZ],
            [d2H_dKX_dKZ,   d2H_dKY_dKZ,   d2H_dKZ2   ]
        ]))

    return grad_grad_H, gradK_grad_H, gradK_gradK_H



def convert_hessians(
    q_start: FloatArray,
    dH: dict,
    grad_grad_H_start: FloatArray,
    gradK_grad_H_start: FloatArray,
    gradK_gradK_H_start: FloatArray,
    start: Literal["cartesian", "cylindrical"],
    end: Literal["cylindrical", "cartesian"]
) -> Tuple[FloatArray, FloatArray, FloatArray]:
    """***TODO from the previous code; not sure if mixed derivatives are correct or not***

    Convert Hessians from Cartesian to cylindrical or vice versa

    Parameters
    ----------
    q_start : (3,) or (3,N) FloatArray
        The position of the corresponding Hessians
    dH : dict
        First order derivatives of the dispersion relation H, i.e. `hamiltonian.derivatives(...)`
    grad_grad_H_start : (3,3) or (N,3,3) FloatArray
        The spatial Hessian
    gradK_grad_H_start : (3,3) or (N,3,3) FloatArray
        The mixed Hessian
    gradK_gradK_H_start : (3,3) or (N,3,3) FloatArray
        The wavevector Hessian
    start : Literal["cartesian", "cylindrical"]
        The coordinate system used for the current Hessian
    end : Literal["cylindrical", "cartesian"]
        The coordinate system to convert the current Hessian into
    """

    if grad_grad_H_start.shape == (3,3):
        squeeze = True
        grad_grad_H_start = np.array([grad_grad_H_start])
        gradK_grad_H_start = np.array([gradK_grad_H_start])
        gradK_gradK_H_start = np.array([gradK_gradK_H_start])
    else:
        squeeze = False

    ij = np.triu_indices(3)
    
    if start.lower() == "cylindrical" and end.lower() == "cartesian":
        R, zeta, _ = q_start
        R2 = R**2
        R3 = R**3
        sin = np.sin(zeta)
        cos = np.cos(zeta)
        sincos = sin*cos
        sin2 = sin**2
        cos2 = cos**2

        # KR, Kzeta, KZ = 

        dH_dR = dH["dH_dR"]
        dH_dzeta = np.zeros_like(dH["dH_dR"])
        dH_dKR = dH["dH_dKR"]
        dH_dKzeta = dH["dH_dKzeta"]

        d2H_dR2, d2H_dR_dzeta, d2H_dR_dZ, d2H_dzeta2, d2H_dzeta_dZ, d2H_dZ2 = grad_grad_H_start[:, ij[0], ij[1]].T
        d2H_dR_dKR, d2H_dzeta_dKR, d2H_dZ_dKR, d2H_dR_dKzeta, d2H_dzeta_dKzeta, d2H_dZ_dKzeta, d2H_dR_dKZ, d2H_dzeta_dKZ, d2H_dZ_dKZ = gradK_grad_H_start.reshape(gradK_grad_H_start.shape[0], 9).T
        d2H_dKR2, d2H_dKR_dKzeta, d2H_dKR_dKZ, d2H_dKzeta2, d2H_dKzeta_dKZ, d2H_dKZ2 = gradK_gradK_H_start[:, ij[0], ij[1]].T

        # TODO from the previous code; not sure if mixed derivatives are correct or not
        d2H_dX2   = cos2*d2H_dR2 - 2*sincos/R*d2H_dR_dzeta + sin2/R2*d2H_dzeta2 + sin2/R*dH_dR + 2*sincos/R2*dH_dzeta
        d2H_dX_dY = sincos*d2H_dR2 + (cos2-sin2)/R*d2H_dR_dzeta - sincos/R2*d2H_dzeta2 - sincos/R*dH_dR + (sin2-cos2)/R2*dH_dzeta
        d2H_dX_dZ = cos*d2H_dR_dZ - sin/R*d2H_dzeta_dZ
        d2H_dY2   = sin2*d2H_dR2 + 2*sincos/R*d2H_dR_dzeta + cos2/R2*d2H_dzeta2 + cos2/R*dH_dR - 2*sincos/R2*dH_dzeta
        d2H_dY_dZ = sin*d2H_dR_dZ + cos/R*d2H_dzeta_dZ
        d2H_dZ2   = d2H_dZ2
        d2H_dX_dKx = sin2/R2*d2H_dzeta2 + cos2*d2H_dR_dKR - sincos/R*d2H_dzeta_dKR - sincos*d2H_dR_dKzeta + sin2/R*dH_dKR + sincos/R2*dH_dzeta
        d2H_dX_dKy = sincos*d2H_dR_dKR - sin2/R*d2H_dzeta_dKR + R*cos2*d2H_dR_dKzeta - sincos*d2H_dzeta_dKzeta - sincos/R*dH_dKR + dH_dzeta
        d2H_dX_dKz = cos*d2H_dR_dKZ - sin/R*d2H_dzeta_dKZ
        d2H_dY_dKx = sincos*d2H_dR_dKR + cos2/R*d2H_dzeta_dKR - R*sin2*d2H_dR_dKzeta - sincos*d2H_dzeta_dKzeta - sincos/R*dH_dKR - dH_dKzeta
        d2H_dY_dKy = sin2*d2H_dR_dKR + sincos/R*d2H_dzeta_dKR + R*sincos*d2H_dR_dKzeta + cos2*d2H_dzeta_dKzeta + cos2/R*dH_dKR
        d2H_dY_dKz = sin*d2H_dR_dKZ + cos/R*d2H_dzeta_dKZ
        d2H_dZ_dKx = cos*d2H_dZ_dKR - R*sin*d2H_dZ_dKzeta
        d2H_dZ_dKy = sin*d2H_dZ_dKR + R*cos*d2H_dZ_dKzeta
        d2H_dZ_dKz = d2H_dZ_dKZ
        d2H_dKx2    = cos2*d2H_dKR2 - 2*R*sincos*d2H_dKR_dKzeta + R2*sin2*d2H_dKzeta2
        d2H_dKx_dKy = sincos*d2H_dKR2 + R*(cos2-sin2)*d2H_dKR_dKzeta - R2*sincos*d2H_dKzeta2
        d2H_dKx_dKz = cos*d2H_dKR_dKZ - R*sin*d2H_dKzeta_dKZ
        d2H_dKy2    = sin2*d2H_dKR2 + 2*R*sincos*d2H_dKR_dKzeta + R2*cos2*d2H_dKzeta2
        d2H_dKy_dKz = sin*d2H_dKR_dKZ + R*cos*d2H_dKzeta_dKZ
        d2H_dKz2    = d2H_dKZ2

        # first-derivative coefficients: dH/dX = xR*H_R + xz*H_zeta + xK*H_KR + xk*H_Kzeta, etc.
        # xR, xz, xK, xk = cos, -sin/R, -sin*Kzeta/R2,  Ky
        # yR, yz, yK, yk = sin,  cos/R,  cos*Kzeta/R2, -Kx
        # kxK, kxk = cos, -R*sin
        # kyK, kyk = sin,  R*cos

        # d2H_dX2 = (
        #       xR*xR*d2H_dR2 + 2*xR*xz*d2H_dR_dzeta + 2*xR*xK*d2H_dR_dKR + 2*xR*xk*d2H_dR_dKzeta
        #     + xz*xz*d2H_dzeta2 + 2*xz*xK*d2H_dzeta_dKR + 2*xz*xk*d2H_dzeta_dKzeta
        #     + xK*xK*d2H_dKR2 + 2*xK*xk*d2H_dKR_dKzeta + xk*xk*d2H_dKzeta2
        #     + sin2/R*dH_dR + 2*sincos/R2*dH_dzeta
        #     + (2*sincos*Kzeta/R3 - sin2*KR/R2)*dH_dKR)

        # d2H_dX_dY = (xR*yR*d2H_dR2 + (xR*yz + xz*yR)*d2H_dR_dzeta
        #             + (xR*yK + xK*yR)*d2H_dR_dKR + (xR*yk + xk*yR)*d2H_dR_dKzeta
        #             + xz*yz*d2H_dzeta2
        #             + (xz*yK + xK*yz)*d2H_dzeta_dKR + (xz*yk + xk*yz)*d2H_dzeta_dKzeta
        #             + xK*yK*d2H_dKR2 + (xK*yk + xk*yK)*d2H_dKR_dKzeta + xk*yk*d2H_dKzeta2
        #             - sincos/R*dH_dR + (sin2 - cos2)/R2*dH_dzeta
        #             + (sincos*KR/R2 + (sin2 - cos2)*Kzeta/R3)*dH_dKR)

        # d2H_dY2 = (yR*yR*d2H_dR2 + 2*yR*yz*d2H_dR_dzeta + 2*yR*yK*d2H_dR_dKR + 2*yR*yk*d2H_dR_dKzeta
        #         + yz*yz*d2H_dzeta2 + 2*yz*yK*d2H_dzeta_dKR + 2*yz*yk*d2H_dzeta_dKzeta
        #         + yK*yK*d2H_dKR2 + 2*yK*yk*d2H_dKR_dKzeta + yk*yk*d2H_dKzeta2
        #         + cos2/R*dH_dR - 2*sincos/R2*dH_dzeta
        #         - (cos2*KR/R2 + 2*sincos*Kzeta/R3)*dH_dKR)

        # d2H_dX_dZ = xR*d2H_dR_dZ + xz*d2H_dzeta_dZ + xK*d2H_dZ_dKR + xk*d2H_dZ_dKzeta
        # d2H_dY_dZ = yR*d2H_dR_dZ + yz*d2H_dzeta_dZ + yK*d2H_dZ_dKR + yk*d2H_dZ_dKzeta
        # # d2H_dZ2 unchanged

        # d2H_dX_dKx = (xR*kxK*d2H_dR_dKR + xR*kxk*d2H_dR_dKzeta
        #             + xz*kxK*d2H_dzeta_dKR + xz*kxk*d2H_dzeta_dKzeta
        #             + xK*kxK*d2H_dKR2 + (xK*kxk + xk*kxK)*d2H_dKR_dKzeta + xk*kxk*d2H_dKzeta2
        #             + sin2/R*dH_dKR)

        # d2H_dX_dKy = (xR*kyK*d2H_dR_dKR + xR*kyk*d2H_dR_dKzeta
        #             + xz*kyK*d2H_dzeta_dKR + xz*kyk*d2H_dzeta_dKzeta
        #             + xK*kyK*d2H_dKR2 + (xK*kyk + xk*kyK)*d2H_dKR_dKzeta + xk*kyk*d2H_dKzeta2
        #             - sincos/R*dH_dKR + dH_dKzeta)

        # d2H_dY_dKx = (yR*kxK*d2H_dR_dKR + yR*kxk*d2H_dR_dKzeta
        #             + yz*kxK*d2H_dzeta_dKR + yz*kxk*d2H_dzeta_dKzeta
        #             + yK*kxK*d2H_dKR2 + (yK*kxk + yk*kxK)*d2H_dKR_dKzeta + yk*kxk*d2H_dKzeta2
        #             - sincos/R*dH_dKR - dH_dKzeta)

        # d2H_dY_dKy = (yR*kyK*d2H_dR_dKR + yR*kyk*d2H_dR_dKzeta
        #             + yz*kyK*d2H_dzeta_dKR + yz*kyk*d2H_dzeta_dKzeta
        #             + yK*kyK*d2H_dKR2 + (yK*kyk + yk*kyK)*d2H_dKR_dKzeta + yk*kyk*d2H_dKzeta2
        #             + cos2/R*dH_dKR)

        # d2H_dX_dKz = xR*d2H_dR_dKZ + xz*d2H_dzeta_dKZ + xK*d2H_dKR_dKZ + xk*d2H_dKzeta_dKZ
        # d2H_dY_dKz = yR*d2H_dR_dKZ + yz*d2H_dzeta_dKZ + yK*d2H_dKR_dKZ + yk*d2H_dKzeta_dKZ

        # d2H_dZ_dKx = cos*d2H_dZ_dKR - R*sin*d2H_dZ_dKzeta
        # d2H_dZ_dKy = sin*d2H_dZ_dKR + R*cos*d2H_dZ_dKzeta
        # d2H_dZ_dKz = d2H_dZ_dKZ

        # d2H_dKx2    = cos2*d2H_dKR2 - 2*R*sincos*d2H_dKR_dKzeta + R2*sin2*d2H_dKzeta2
        # d2H_dKx_dKy = sincos*d2H_dKR2 + R*(cos2 - sin2)*d2H_dKR_dKzeta - R2*sincos*d2H_dKzeta2
        # d2H_dKy2    = sin2*d2H_dKR2 + 2*R*sincos*d2H_dKR_dKzeta + R2*cos2*d2H_dKzeta2
        # d2H_dKx_dKz = cos*d2H_dKR_dKZ - R*sin*d2H_dKzeta_dKZ
        # d2H_dKy_dKz = sin*d2H_dKR_dKZ + R*cos*d2H_dKzeta_dKZ
        # d2H_dKz2    = d2H_dKZ2

        new_grad_grad_H = reshape_33N_to_N33(np.array([
            [d2H_dX2,       d2H_dX_dY,     d2H_dX_dZ],
            [d2H_dX_dY,     d2H_dY2,       d2H_dY_dZ],
            [d2H_dX_dZ,     d2H_dY_dZ,     d2H_dZ2  ]
        ]))
        new_gradK_grad_H = reshape_33N_to_N33(np.array([
            [d2H_dX_dKx,    d2H_dY_dKx,    d2H_dZ_dKx],
            [d2H_dX_dKy,    d2H_dY_dKy,    d2H_dZ_dKy],
            [d2H_dX_dKz,    d2H_dY_dKz,    d2H_dZ_dKz]
        ]))
        new_gradK_gradK_H = reshape_33N_to_N33(np.array([
            [d2H_dKx2,      d2H_dKx_dKy,   d2H_dKx_dKz],
            [d2H_dKx_dKy,   d2H_dKy2,      d2H_dKy_dKz],
            [d2H_dKx_dKz,   d2H_dKy_dKz,   d2H_dKz2   ]
        ]))

    elif start.lower() == "cartesian" and end.lower() == "cylindrical": raise RuntimeError("Hessian conversion from cart. to cyld. not implemented yet")
    else: raise ValueError(f"Hessian conversion from {start} to {end} not supported")

    if squeeze: return np.squeeze(new_grad_grad_H), np.squeeze(new_gradK_grad_H), np.squeeze(new_gradK_gradK_H)
    else:       return new_grad_grad_H, new_gradK_grad_H, new_gradK_gradK_H

        # original code, from eduard_testfile.ipynb

        # scotty_2d_data["d2H_dX2"]   = _cos2*_d2H_dR2 - 2*_sincos/_R*_d2H_dR_dzeta + _sin2/_R2*_d2H_dzeta2 + _sin2/_R*_dH_dR + 2*_sincos/_R2*_dH_dzeta
        # scotty_2d_data["d2H_dX_dY"] = _sincos*_d2H_dR2 + (_cos2-_sin2)/_R*_d2H_dR_dzeta - _sincos/_R2*_d2H_dzeta2 - _sincos/_R*_dH_dR + (_sin2-_cos2)/_R2*_dH_dzeta
        # scotty_2d_data["d2H_dX_dZ"] = _cos*_d2H_dR_dZ - _sin/_R*_d2H_dzeta_dZ
        # scotty_2d_data["d2H_dY2"]   = _sin2*_d2H_dR2 + 2*_sincos/_R*_d2H_dR_dzeta + _cos2/_R2*_d2H_dzeta2 + _cos2/_R*_dH_dR - 2*_sincos/_R2*_dH_dzeta
        # scotty_2d_data["d2H_dY_dZ"] = _sin*_d2H_dR_dZ + _cos/_R*_d2H_dzeta_dZ
        # scotty_2d_data["d2H_dZ2"]   = _d2H_dZ2

        # scotty_2d_data["d2H_dX_dKx"] = _sin2/_R2*_d2H_dzeta2 + _cos2*_d2H_dKR_dR - _sincos/_R*_d2H_dKR_dzeta - _sincos*_d2H_dKzeta_dR + _sin2/_R*_dH_dKR + _sincos/_R2*_dH_dzeta
        # scotty_2d_data["d2H_dX_dKy"] = _sincos*_d2H_dKR_dR - _sin2/_R*_d2H_dKR_dzeta + _R*_cos2*_d2H_dKzeta_dR - _sincos*_d2H_dKzeta_dzeta - _sincos/_R*_dH_dKR + _dH_dzeta
        # scotty_2d_data["d2H_dX_dKz"] = _cos*_d2H_dKZ_dR - _sin/_R*_d2H_dKZ_dzeta
        # scotty_2d_data["d2H_dY_dKx"] = _sincos*_d2H_dKR_dR + _cos2/_R*_d2H_dKR_dzeta - _R*_sin2*_d2H_dKzeta_dR - _sincos*_d2H_dKzeta_dzeta - _sincos/_R*_dH_dKR - _dH_dKzeta
        # scotty_2d_data["d2H_dY_dKy"] = _sin2*_d2H_dKR_dR + _sincos/_R*_d2H_dKR_dzeta + _R*_sincos*_d2H_dKzeta_dR + _cos2*_d2H_dKzeta_dzeta + _cos2/_R*_dH_dKR
        # scotty_2d_data["d2H_dY_dKz"] = _sin*_d2H_dKZ_dR + _cos/_R*_d2H_dKZ_dzeta
        # scotty_2d_data["d2H_dZ_dKx"] = _cos*_d2H_dKR_dZ - _R*_sin*_d2H_dKzeta_dZ
        # scotty_2d_data["d2H_dZ_dKy"] = _sin*_d2H_dKR_dZ + _R*_cos*_d2H_dKzeta_dZ
        # scotty_2d_data["d2H_dZ_dKz"] = _d2H_dKZ_dZ

        # scotty_2d_data["d2H_dKx2"]    = _cos2*_d2H_dKR2 - 2*_R*_sincos*_d2H_dKR_dKzeta + _R2*_sin2*_d2H_dKzeta2
        # scotty_2d_data["d2H_dKx_dKy"] = _sincos*_d2H_dKR2 + _R*(_cos2-_sin2)*_d2H_dKR_dKzeta - _R2*_sincos*_d2H_dKzeta2
        # scotty_2d_data["d2H_dKx_dKz"] = _cos*_d2H_dKR_dKZ - _R*_sin*_d2H_dKzeta_dKZ
        # scotty_2d_data["d2H_dKy2"]    = _sin2*_d2H_dKR2 + 2*_R*_sincos*_d2H_dKR_dKzeta + _R2*_cos2*_d2H_dKzeta2
        # scotty_2d_data["d2H_dKy_dKz"] = _sin*_d2H_dKR_dKZ + _R*_cos*_d2H_dKzeta_dKZ
        # scotty_2d_data["d2H_dKz2"]    = _d2H_dKZ2