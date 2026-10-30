from __future__ import annotations
import logging
import numpy as np
from scipy import constants

# from scotty.checks import VALID_LAUNCH_MODE_FLAGS
from scotty.logger import arr2str
from scotty.typing import ArrayLike, Array, FloatArray, ComplexFloatArray
from typing import Callable, Union, Tuple, Optional, List, Literal, TextIO, cast

log = logging.getLogger(__name__)

##################################################
#
# UNIT CONVERSIONS
#
##################################################


def freq_GHz_to_angular_frequency(freq_GHz: float) -> float:
    """
    Convert frequency to angular frequency

    Parameters
    ----------
    freq_GHz : float
        Frequency in gigahertz (GHz)

    Returns
    ----------
    output : float
        Angular frequency in radians per second (rad/s)
    """
    return 2 * np.pi * 1e9 * freq_GHz


def angular_frequency_to_wavenumber(angular_frequency: float) -> float:
    """
    Convert angular frequency to wavenumber

    Parameters
    ----------
    angular_frequency : float
        Angular frequency (rad/s)

    Returns
    ----------
    output : float
        Wavenumber in m^-1
    """
    return angular_frequency / constants.c


def freq_GHz_to_wavenumber(freq_GHz: float) -> float:
    """
    Convert frequency to wavenumber

    Parameters
    ----------
    freq_GHz : float
        Frequency in gigahertz (GHz)

    Returns
    ----------
    output : float
        Wavenumber in m^-1
    """
    return angular_frequency_to_wavenumber(freq_GHz_to_angular_frequency(freq_GHz))


##################################################
#
# COORDINATE TRANSFORMS
#
##################################################


def find_q_labframe_cart_to_cyl(q_lab_cart: FloatArray) -> FloatArray:
    """
    Converts `q_labframe` from Cartesian to cylindrical coordinates, both in
    the lab frame (not the beam frame). The shape of `q_lab_cart` must be
    either `(3,)` or `(3, N)`, where N is the number of points at which `q`
    is evaluated

    Parameters
    ----------
    q_lab_cart : (3,) or (3,N) FloatArray
        Position vector, lab frame, in Cartesian

    Returns
    ----------
    q_lab_cyl : (3,) or (3,N) FloatArray
        Position vector, lab frame, in cylindrical
    """
    if (q_lab_cart.ndim == 1 or q_lab_cart.ndim == 2) and q_lab_cart.shape[0] == 3:
        q_X, q_Y, q_Z = q_lab_cart
    else:
        raise ValueError(
            f"`find_q_labframe_cart_to_cyl`: Expected input to be shape (3,) or (3, N) but got shape {q_lab_cart.shape}"
        )
    return np.array([np.sqrt(q_X**2 + q_Y**2), np.arctan2(q_Y, q_X), q_Z])


def find_q_labframe_cyl_to_cart(q_lab_cyl: FloatArray) -> FloatArray:
    """
    Converts `q_labframe` from cylindrical to Cartesian coordinates, both in
    the lab frame (not the beam frame). The shape of `q_lab_cyl` must be
    either `(3,)` or `(3, N)`, where N is the number of points at which `q`
    is evaluated

    Parameters
    ----------
    q_lab_cyl : (3,) or (3,N) FloatArray
        Position vector, lab frame, in cylindrical

    Returns
    ----------
    q_lab_cart : (3,) or (3,N) FloatArray
        Position vector, lab frame, in Cartesian
    """
    if (q_lab_cyl.ndim == 1 or q_lab_cyl.ndim == 2) and q_lab_cyl.shape[0] == 3:
        q_R, q_p, q_Z = q_lab_cyl
    else:
        raise ValueError(
            f"`find_q_labframe_cyl_to_cart`: Expected input to be shape (3,) or (3, N) but got shape {q_lab_cyl.shape}"
        )
    return np.array([q_R * np.cos(q_p), q_R * np.sin(q_p), q_Z])


def find_K_labframe_cart_to_cyl(
    K_lab_cart: FloatArray, q_lab_cart: FloatArray
) -> FloatArray:
    """
    Converts `K_labframe` from Cartesian to cylindrical coordinates, both in
    the lab frame (not the beam frame). The shape of `K_lab_cart` must be
    either `(3,)` or `(3, N)`, where N is the number of points at which `K`
    is evaluated, and `K_lab_cart` and `q_lab_cart` must have the same shape

    Parameters
    ----------
    K_lab_cart : (3,) or (3,N) FloatArray
        Wavevector, lab frame, in Cartesian
    q_lab_cart : (3,) or (3,N) FloatArray
        Position vector, lab frame, in Cartesian

    Returns
    ----------
    K_lab_cyl : (3,) or (3,N) FloatArray
        Wavevector, lab frame, in cylindrical
    """
    if (q_lab_cart.ndim == 1 or q_lab_cart.ndim == 2) and q_lab_cart.shape[0] == 3:
        q_R, q_zeta, q_Z = find_q_labframe_cart_to_cyl(q_lab_cart=q_lab_cart)
    else:
        raise ValueError(
            f"`find_K_labframe_cart_to_cyl`: Expected input to be shape (3,) or (3, N) but got shape {q_lab_cart.shape}"
        )

    if (K_lab_cart.ndim == 1 or K_lab_cart.ndim == 2) and K_lab_cart.shape[0] == 3:
        K_X, K_Y, K_Z = K_lab_cart
    else:
        raise ValueError(
            f"`find_K_labframe_cart_to_cyl`: Expected input to be shape (3,) or (3, N) but got shape {K_lab_cart.shape}"
        )

    if not K_lab_cart.shape == q_lab_cart.shape:
        raise ValueError(
            f"`find_K_labframe_cart_to_cyl`: Expected `K_lab_cart` and `q_lab_cart` to have the same shape, but got {K_lab_cart.shape} and {q_lab_cart.shape}"
        )

    sin_zeta = np.sin(q_zeta)
    cos_zeta = np.cos(q_zeta)

    return np.array(
        [K_X * cos_zeta + K_Y * sin_zeta, (-K_X * sin_zeta + K_Y * cos_zeta) * q_R, K_Z]
    )


def find_K_labframe_cyl_to_cart(
    K_lab_cyl: FloatArray, q_lab_cyl: FloatArray
) -> FloatArray:
    """
    Converts `K_labframe` from cylindrical to Cartesian coordinates, both in
    the lab frame (not the beam frame). The shape of `K_lab_cyl` must be
    either `(3,)` or `(3, N)`, where N is the number of points at which `K`
    is evaluated, and `K_lab_cyl` and `q_lab_cyl` must have the same shape

    Parameters
    ----------
    K_lab_cyl : (3,) or (3,N) FloatArray
        Wavevector, lab frame, in cylindrical
    q_lab_cyl : (3,) or (3,N) FloatArray
        Position vector, lab frame, in cylindrical

    Returns
    ----------
    K_lab_cart : (3,) or (3,N) FloatArray
        Wavevector, lab frame, in Cartesian
    """
    if (q_lab_cyl.ndim == 1 or q_lab_cyl.ndim == 2) and q_lab_cyl.shape[0] == 3:
        q_R, q_zeta, q_Z = q_lab_cyl
    else:
        raise ValueError(
            f"`find_K_labframe_cyl_to_cart`: Expected input to be shape (3,) or (3, N) but got shape {q_lab_cyl.shape}"
        )

    if (K_lab_cyl.ndim == 1 or K_lab_cyl.ndim == 2) and K_lab_cyl.shape[0] == 3:
        K_R, K_zeta, K_Z = K_lab_cyl
    else:
        raise ValueError(
            f"`find_K_labframe_cyl_to_cart`: Expected input to be shape (3,) or (3, N) but got shape {K_lab_cyl.shape}"
        )

    if not K_lab_cyl.shape == q_lab_cyl.shape:
        raise ValueError(
            f"`find_K_labframe_cyl_to_cart`: Expected `K_lab_cyl` and `q_lab_cyl` to have the same shape, but got {K_lab_cyl.shape} and {q_lab_cyl.shape}"
        )

    sin_zeta = np.sin(q_zeta)
    cos_zeta = np.cos(q_zeta)

    return np.array(
        [
            K_R * cos_zeta - K_zeta * sin_zeta / q_R,
            K_R * sin_zeta + K_zeta * cos_zeta / q_R,
            K_Z,
        ]
    )


def find_Psi_3D_labframe_cart_to_cyl(
    Psi_3D_labframe_cartesian: ComplexFloatArray,
    K_lab_cart: FloatArray,
    q_lab_cart: FloatArray,
) -> ComplexFloatArray:
    """
    Converts `Psi_3D_labframe` from Cartesian to cylindrical coordinates,
    both in the lab frame (not the beam frame). The shape of
    `Psi_3D_labframe_cartesian` must be either `(3, 3)` or `(N, 3, 3)`,
    where N is the number of points at which `Psi` is evaluated, with
    `K_lab_cart` and `q_lab_cart` having the same shape and also compatible
    with `Psi_3D_labframe_cartesian` (i.e. if `Psi_3D_labframe_cartesian` is
    `(3, 3)` then `K_lab_cart` and `q_lab_cart` must be `(3,)`, and if
    `Psi_3D_labframe_cartesian` is `(N, 3, 3)` then `K_lab_cart` and
    `q_lab_cart` must be `(3, N)`)

    Note that the shape of `Psi_3D_labframe_cartesian` is different from the
    shapes of `K_lab_cart` and `q_lab_cart` because of the way we evolve
    the beam

    Parameters
    ----------
    Psi_3D_labframe_cartesian : (3,3) or (N,3,3) FloatArray
        Beam matrix, lab frame, in Cartesian
    K_lab_cart : (3,) or (3,N) FloatArray
        Wavevector, lab frame, in Cartesian
    q_lab_cart : (3,) or (3,N) FloatArray
        Position vector, lab frame, in Cartesian

    Returns
    ----------
    Psi_3D_labframe_cylindrical : (3,3) or (N,3,3) FloatArray
        Beam matrix, lab frame, in cylindrical
    """
    q_X, q_Y, _ = q_lab_cart
    q_R, q_zeta, q_Z = find_q_labframe_cart_to_cyl(q_lab_cart)
    sin_zeta = np.sin(q_zeta)
    cos_zeta = np.cos(q_zeta)
    K_X, K_Y, _ = K_lab_cart
    K_R, K_zeta, K_Z = find_K_labframe_cart_to_cyl(K_lab_cart, q_lab_cart)

    if Psi_3D_labframe_cartesian.ndim == 2:  # A single matrix of Psi
        Psi = np.array([Psi_3D_labframe_cartesian])
        N = 1
        squeeze = True
    elif Psi_3D_labframe_cartesian.ndim == 3:  # A stack of matrices of Psi
        Psi = Psi_3D_labframe_cartesian
        N = Psi.shape[0]
        squeeze = False
    else:
        raise ValueError(
            f"`Psi_3D_labframe_cartesian` has an invalid number of dimensions: Expected 2 or 3 but got ndim = {Psi_3D_labframe_cartesian.ndim}"
        )

    Psi_XX = Psi[:, 0, 0]
    Psi_XY = Psi[:, 0, 1]
    Psi_XZ = Psi[:, 0, 2]
    Psi_YY = Psi[:, 1, 1]
    Psi_YZ = Psi[:, 1, 2]
    Psi_ZZ = Psi[:, 2, 2]

    Psi_cyl = np.zeros([N, 3, 3], dtype="complex128")
    Psi_cyl[:, 0, 0] = (
        Psi_XX * cos_zeta**2 + 2 * Psi_XY * sin_zeta * cos_zeta + Psi_YY * sin_zeta**2
    )
    Psi_cyl[:, 0, 1] = (
        -Psi_XX * sin_zeta * cos_zeta
        + Psi_XY * (cos_zeta**2 - sin_zeta**2)
        + Psi_YY * sin_zeta * cos_zeta
    ) * q_R + K_zeta / q_R
    Psi_cyl[:, 0, 2] = Psi_XZ * cos_zeta + Psi_YZ * sin_zeta
    Psi_cyl[:, 1, 1] = (
        Psi_XX * sin_zeta**2 - 2 * Psi_XY * sin_zeta * cos_zeta + Psi_YY * cos_zeta**2
    ) * q_R**2 - K_R * q_R
    Psi_cyl[:, 1, 2] = (-Psi_XZ * sin_zeta + Psi_YZ * cos_zeta) * q_R
    Psi_cyl[:, 2, 2] = Psi_ZZ
    Psi_cyl[:, 1, 0] = Psi_cyl[:, 0, 1]
    Psi_cyl[:, 2, 0] = Psi_cyl[:, 0, 2]
    Psi_cyl[:, 2, 1] = Psi_cyl[:, 1, 2]

    return Psi_cyl[0] if squeeze else Psi_cyl


def find_Psi_3D_labframe_cyl_to_cart(
    Psi_3D_labframe_cylindrical: ComplexFloatArray,
    K_lab_cyl: FloatArray,
    q_lab_cyl: FloatArray,
) -> ComplexFloatArray:
    """
    Converts `Psi_3D_labframe` from cylindrical to Cartesian coordinates,
    both in the lab frame (not the beam frame). The shape of
    `Psi_3D_labframe_cylindrical` must be either `(3, 3)` or `(N, 3, 3)`,
    where N is the number of points at which `Psi` is evaluated, with
    `K_lab_cart` and `q_lab_cart` having the same shape and also compatible
    with `Psi_3D_labframe_cylindrical` (i.e. if `Psi_3D_labframe_cylindrical`
    is `(3, 3)` then `K_lab_cyl` and `q_lab_cyl` must be `(3,)`, and if
    `Psi_3D_labframe_cylindrical` is `(N, 3, 3)` then `K_lab_cyl` and
    `q_lab_cyl` must be `(3, N)`)

    Note that the shape of `Psi_3D_labframe_cylindrical` is different from
    the shapes of `K_lab_cyl` and `q_lab_cyl` because of the way we evolve
    the beam

    Parameters
    ----------
    Psi_3D_labframe_cylindrical : (3,3) or (N,3,3) FloatArray
        Beam matrix, lab frame, in cylindrical
    K_lab_cyl : (3,) or (3,N) FloatArray
        Wavevector, lab frame, in cylindrical
    q_lab_cyl : (3,) or (3,N) FloatArray
        Position vector, lab frame, in cylindrical

    Returns
    ----------
    Psi_3D_labframe_cartesian : (3,3) or (N,3,3) FloatArray
        Beam matrix, lab frame, in Cartesian
    """
    q_R, q_zeta, q_Z = q_lab_cyl
    q_X, q_Y, _ = find_q_labframe_cyl_to_cart(q_lab_cyl)
    K_R, K_zeta, K_Z = K_lab_cyl
    # Treat a single point as N = 1, so the rotation matrix below is always
    # built as (N,3,3) rather than a transposed (3,3)
    q_R, q_zeta, K_R, K_zeta = np.atleast_1d(q_R, q_zeta, K_R, K_zeta)

    if Psi_3D_labframe_cylindrical.ndim == 2:  # A single matrix of Psi
        Psi = np.array([Psi_3D_labframe_cylindrical])
        N = 1
        squeeze = True
    elif Psi_3D_labframe_cylindrical.ndim == 3:  # A stack of matrices of Psi
        Psi = Psi_3D_labframe_cylindrical
        N = Psi.shape[0]
        squeeze = False
    else:
        raise ValueError(
            f"`Psi_3D_labframe_cylindrical` has an invalid number of dimensions: Expected 2 or 3 but got ndim = {Psi_3D_labframe_cylindrical.ndim}"
        )

    Psi_RR = Psi[:, 0, 0]
    Psi_Rzeta = Psi[:, 0, 1]
    Psi_RZ = Psi[:, 0, 2]
    Psi_zetazeta = Psi[:, 1, 1]
    Psi_zetaZ = Psi[:, 1, 2]
    Psi_ZZ = Psi[:, 2, 2]

    Psi_temp = np.zeros([N, 3, 3], dtype="complex128")
    Psi_temp[:, 0, 0] = Psi_RR
    Psi_temp[:, 0, 1] = Psi_Rzeta / q_R - K_zeta / q_R**2
    Psi_temp[:, 0, 2] = Psi_RZ
    Psi_temp[:, 1, 1] = Psi_zetazeta / q_R**2 + K_R / q_R
    Psi_temp[:, 1, 2] = Psi_zetaZ / q_R
    Psi_temp[:, 2, 2] = Psi_ZZ
    Psi_temp[:, 1, 0] = Psi_temp[:, 0, 1]
    Psi_temp[:, 2, 0] = Psi_temp[:, 0, 2]
    Psi_temp[:, 2, 1] = Psi_temp[:, 1, 2]

    sin_zeta = np.sin(q_zeta)
    cos_zeta = np.cos(q_zeta)
    zeros = np.zeros_like(q_zeta)
    ones = np.ones_like(q_zeta)

    rotation_matrix_xi = np.moveaxis(
        np.array(
            [
                [cos_zeta, -sin_zeta, zeros],
                [sin_zeta, cos_zeta, zeros],
                [zeros, zeros, ones],
            ]
        ),
        -1,
        0,
    )  # moveaxis to for broadcasting as (N,3,3)

    # Rotation matrices are orthogonal, so the inverse is the transpose of
    # each (3,3) matrix
    rotation_matrix_xi_inverse = np.swapaxes(rotation_matrix_xi, -1, -2)

    Psi_cart = np.matmul(
        np.matmul(rotation_matrix_xi, Psi_temp), rotation_matrix_xi_inverse
    )

    return Psi_cart[0] if squeeze else Psi_cart


def find_vector_and_q_cyl_to_cart(
    vector_labframe_cyl: FloatArray, q_labframe_cyl: FloatArray
) -> Tuple[FloatArray, FloatArray]:
    """
    Converts `vector_labframe_cyl` and `q_labframe_cyl` from cylindrical
    to Cartesian

    Parameters
    ----------
    vector_labframe_cyl : (3,) FloatArray
        Vector in cylindrical; note that this function cannot be used to
        convert the wavevector, because the `K_zeta` component is the
        unitless toroidal mode number
    q_labframe_cyl : (3,) FloatArray
        Position vector in cylindrical

    Returns
    ----------
    vector_labframe_cart : (3,) FloatArray
        Vector in Cartesian
    q_labframe_cart : (3,) FloatArray
        Position vector in Cartesian
    """
    if not vector_labframe_cyl.shape == q_labframe_cyl.shape:
        raise ValueError(
            f"`vector_labframe_cyl` and `q_labframe_cyl` must have the same shape, but got {vector_labframe_cyl.shape} and {q_labframe_cyl.shape}"
        )
    elif not vector_labframe_cyl.shape[0] == 3:
        raise ValueError(f"`vector_labframe_cyl` must be indexed as a 3-vector")
    elif not q_labframe_cyl.shape[0] == 3:
        raise ValueError(f"`q_labframe_cyl` must be indexed as a 3-vector")

    v_R, v_zeta, v_Z = vector_labframe_cyl
    q_R, q_zeta, q_Z = q_labframe_cyl
    sin_zeta = np.sin(q_zeta)
    cos_zeta = np.cos(q_zeta)

    q_labframe_cart = np.zeros_like(q_labframe_cyl, dtype=q_labframe_cyl.dtype)
    q_labframe_cart[0] = q_R * cos_zeta
    q_labframe_cart[1] = q_R * sin_zeta
    q_labframe_cart[2] = q_Z

    v_labframe_cart = np.zeros_like(
        vector_labframe_cyl, dtype=vector_labframe_cyl.dtype
    )
    v_labframe_cart[0] = v_R * cos_zeta - v_zeta * sin_zeta
    v_labframe_cart[1] = v_R * sin_zeta + v_zeta * cos_zeta
    v_labframe_cart[2] = v_Z

    return v_labframe_cart, q_labframe_cart


##################################################
#
# PLASMA CALCULATIONS
#
##################################################


def find_electron_mass(temperature: Optional[ArrayLike] = None) -> ArrayLike:
    """
    Finds the electron mass using Mazzucato's relativistic corrections

    Parameters
    ----------
    temperature : ArrayLike or None
        Temperature, in keV, at the points in the plasma to be queried

    Returns
    ----------
    output : ArrayLike
        If `temperature` is `ArrayLike`, then return the corrected electron
        mass; otherwise just return the normal electron mass from
        `scipy.constants.m_e`
    """
    # Mazzucato's relativistic correction
    factor = 1 if temperature is None else 1 + temperature * 4.892 * 10 ** (-3)
    return factor * constants.m_e


def find_normalised_plasma_ang_freq(
    launch_angular_freq: float,
    electron_density: ArrayLike,
    temperature: Optional[ArrayLike] = None,
) -> ArrayLike:
    """
    Finds the plasma frequency, normalised to the angular frequency at launch

    Parameters
    ----------
    launch_angular_freq : float
        Angular frequency, in rad/s, at launch
    electron_density : ArrayLike
        Electron density, in 10^19 m^-3, at the points in the plasma to be
        queried
    temperature : ArrayLike or None
        Temperature, in keV, at the points in the plasma to be queried

    Returns
    ----------
    output : ArrayLike
        Normalised plasma frequency at the queried points in the plasma
    """
    m_e = find_electron_mass(temperature)
    return (
        constants.e * np.sqrt(electron_density * 10**19 / (constants.epsilon_0 * m_e))
    ) / launch_angular_freq


def find_normalised_gyro_ang_freq(
    launch_angular_freq: float,
    B_total: ArrayLike,
    temperature: Optional[ArrayLike] = None,
) -> ArrayLike:
    """
    Finds the gyrofrequency, normalised to the angular frequency at launch

    Parameters
    ----------
    launch_angular_freq : float
        Angular frequency, in rad/s, at launch
    B_total : ArrayLike
        Magnitude of the magnetic field, in Tesla, at the points in the
        plasma to be queried
    temperature : ArrayLike or None
        Temperature, in keV, at the points in the plasma to be queried

    Returns
    ----------
    output : ArrayLike
        Normalised gyrofrequency at the queried points in the plasma
    """
    m_e = find_electron_mass(temperature)
    return constants.e * B_total / (m_e * launch_angular_freq)


def find_normalised_angular_freqs(
    launch_angular_freq: float,
    B_total: ArrayLike,
    electron_density: ArrayLike,
    temperature: Optional[ArrayLike] = None,
) -> List[ArrayLike]:
    """
    Finds the electron plasma (angular) frequency, electron cyclotron
    (angular) frequency, left-hand cutoff (angular) frequency, right-hand
    cutoff (angular) frequency, and upper hybrid (angular) frequency, all
    normalised to the angular frequency at launch

    Parameters
    ----------
    launch_angular_freq : float
        Angular frequency, in rad/s, at launch
    B_total : ArrayLike
        Magnitude of the magnetic field, in Tesla, at the points in the
        plasma to be queried
    electron_density : ArrayLike
        Electron density, in 10^19 m^-3, at the points in the plasma to be
        queried
    temperature : ArrayLike or None
        Temperature, in keV, at the points in the plasma to be queried

    Returns
    ----------
    w_pe : ArrayLike
        Normalised plasma frequency at the queried points in the plasma
    w_ce : ArrayLike
        Normalised gyrofrequency at the queried points in the plasma
    w_LH : ArrayLike
        Normalised left hand cutoff frequency at the queried points in the
        plasma
    w_RH : ArrayLike
        Normalised right hand cutoff frequency at the queried points in the
        plasma
    w_UH : ArrayLike
        Normalised upper hybrid frequency at the queried points in the
        plasma
    """
    w_pe = find_normalised_plasma_ang_freq(
        launch_angular_freq, electron_density, temperature
    )
    w_ce = find_normalised_gyro_ang_freq(launch_angular_freq, B_total, temperature)
    w_LH = 0.5 * (-w_ce + np.sqrt(w_ce**2 + 4 * w_pe**2))
    w_RH = 0.5 * (w_ce + np.sqrt(w_ce**2 + 4 * w_pe**2))
    w_UH = np.sqrt(w_pe**2 + w_ce**2)
    return [w_pe, w_ce, w_LH, w_RH, w_UH]


def find_epsilon_terms(
    launch_angular_freq: float,
    B_total: ArrayLike,
    electron_density: ArrayLike,
    temperature: Optional[ArrayLike] = None,
    para: bool = True,
    perp: bool = True,
    g: bool = True,
) -> List[ArrayLike]:
    """
    Finds `epsilon_para` (`epsilon_bb`), `epsilon_perp` (`epsilon_11`), and
    `epsilon_g` (`epsilon_12`) if all flags are `True`

    Parameters
    ----------
    launch_angular_freq : float
        Angular frequency, in rad/s, at launch
    B_total : ArrayLike
        Magnitude of the magnetic field, in Tesla, at the points in the
        plasma to be queried
    electron_density : ArrayLike
        Electron density, in 10^19 m^-3, at the points in the plasma to be
        queried
    temperature : ArrayLike or None
        Temperature, in keV, at the points in the plasma to be queried
    para : bool
        If `True`, return epsilon_para (epsilon_bb)
    perp : bool
        If `True`, return epsilon_perp (epsilon_11)
    g : bool
        If `True`, return epsilon_g (epsilon_12)

    Returns
    ----------
    output : list of ArrayLike
        If all flags are `True`, the entries of the list correspond to:

            epsilon_para : ArrayLike
                Also called epsilon_bb
            epsilon_perp : ArrayLike
                Also called epsilon_11
            epsilon_g : ArrayLike
                Also called epsilon_12

        Otherwise, the corresponding term is omitted but the ordering
        is still the same
    """
    toreturn = []
    normalised_plasma_ang_freq = find_normalised_plasma_ang_freq(
        launch_angular_freq, electron_density, temperature
    )
    normalised_gyro_ang_freq = find_normalised_gyro_ang_freq(
        launch_angular_freq, B_total, temperature
    )

    if para:
        toreturn.append(1 - normalised_plasma_ang_freq**2)
    if perp:
        toreturn.append(
            1 - normalised_plasma_ang_freq**2 / (1 - normalised_gyro_ang_freq**2)
        )
    if g:
        toreturn.append(
            normalised_plasma_ang_freq**2
            * normalised_gyro_ang_freq
            / (1 - normalised_gyro_ang_freq**2)
        )
    return toreturn


def find_epsilon_para(
    launch_angular_freq: float,
    electron_density: ArrayLike,
    temperature: Optional[ArrayLike] = None,
) -> ArrayLike:
    """
    Finds `epsilon_para`. Also called `epsilon_bb` in the paper

    Parameters
    ----------
    launch_angular_freq : float
        Angular frequency, in rad/s, at launch
    electron_density : ArrayLike
        Electron density, in 10^19 m^-3, at the points in the plasma to be
        queried
    temperature : ArrayLike or None
        Temperature, in keV, at the points in the plasma to be queried

    Returns
    ----------
    epsilon_para : ArrayLike
        Also called epsilon_bb
    """
    return find_epsilon_terms(
        launch_angular_freq,
        0,
        electron_density,
        temperature,
        para=True,
        perp=False,
        g=False,
    )[0]


def find_epsilon_perp(
    launch_angular_freq: float,
    B_total: ArrayLike,
    electron_density: ArrayLike,
    temperature: Optional[ArrayLike] = None,
) -> ArrayLike:
    """
    Finds `epsilon_perp`. Also called `epsilon_11` in the paper

    Parameters
    ----------
    launch_angular_freq : float
        Angular frequency, in rad/s, at launch
    B_total : ArrayLike
        Magnitude of the magnetic field, in Tesla, at the points in the
        plasma to be queried
    electron_density : ArrayLike
        Electron density, in 10^19 m^-3, at the points in the plasma to be
        queried
    temperature : ArrayLike or None
        Temperature, in keV, at the points in the plasma to be queried

    Returns
    ----------
    epsilon_perp : ArrayLike
        Also called epsilon_11
    """
    return find_epsilon_terms(
        launch_angular_freq,
        B_total,
        electron_density,
        temperature,
        para=False,
        perp=True,
        g=False,
    )[0]


def find_epsilon_g(
    launch_angular_freq: float,
    B_total: ArrayLike,
    electron_density: ArrayLike,
    temperature: Optional[ArrayLike] = None,
) -> ArrayLike:
    """
    Finds `epsilon_g`. Also called `epsilon_12` in the paper

    Parameters
    ----------
    launch_angular_freq : float
        Angular frequency, in rad/s, at launch
    B_total : ArrayLike
        Magnitude of the magnetic field, in Tesla, at the points in the
        plasma to be queried
    electron_density : ArrayLike
        Electron density, in 10^19 m^-3, at the points in the plasma to be
        queried
    temperature : ArrayLike or None
        Temperature, in keV, at the points in the plasma to be queried

    Returns
    ----------
    epsilon_g : ArrayLike
        Also called epsilon_12
    """
    return find_epsilon_terms(
        launch_angular_freq,
        B_total,
        electron_density,
        temperature,
        para=False,
        perp=False,
        g=True,
    )[0]


def find_Booker_terms(
    launch_angular_freq: float,
    B_total: ArrayLike,
    sin_theta_m_sq: ArrayLike,
    electron_density: ArrayLike,
    temperature: Optional[ArrayLike] = None,
    alpha: bool = True,
    beta: bool = True,
    gamma: bool = True,
) -> List[ArrayLike]:
    """
    Finds `Booker_alpha`, `Booker_beta`, and `Booker_gamma` if all flags
    are `True`

    Parameters
    ----------
    launch_angular_freq : float
        Angular frequency, in rad/s, at launch
    B_total : ArrayLike
        Magnitude of the magnetic field, in Tesla, at the points in the
        plasma to be queried
    sin_theta_m_sq : ArrayLike
        ( sin(theta_m) )^2 at the points in the plasma to be queried
    electron_density : ArrayLike
        Electron density, in 10^19 m^-3, at the points in the plasma to be
        queried
    temperature : ArrayLike or None
        Temperature, in keV, at the points in the plasma to be queried
    alpha : bool
        If `True`, return Booker_alpha
    beta : bool
        If `True`, return Booker_beta
    gamma : bool
        If `True`, return Booker_gamma

    Returns
    ----------
    output : list of ArrayLike
        If all flags are `True`, the entries of the list correspond to:

            Booker_alpha : ArrayLike
                Booker_alpha
            Booker_beta : ArrayLike
                Booker_beta
            Booker_gamma : ArrayLike
                Booker_gamma

        Otherwise, the corresponding term is omitted but the ordering
        is still the same
    """
    toreturn = []
    epsilon_para, epsilon_perp, epsilon_g = find_epsilon_terms(
        launch_angular_freq, B_total, electron_density, temperature
    )

    if alpha:
        toreturn.append(
            epsilon_para * sin_theta_m_sq + epsilon_perp * (1 - sin_theta_m_sq)
        )
    if beta:
        toreturn.append(
            -epsilon_perp * epsilon_para * (1 + sin_theta_m_sq)
            - (epsilon_perp**2 - epsilon_g**2) * (1 - sin_theta_m_sq)
        )
    if gamma:
        toreturn.append(epsilon_para * (epsilon_perp**2 - epsilon_g**2))
    return toreturn


def find_Booker_discriminant(
    launch_angular_freq: float,
    B_total: ArrayLike,
    sin_theta_m_sq: ArrayLike,
    electron_density: ArrayLike,
    temperature: Optional[ArrayLike] = None,
) -> ArrayLike:
    r"""
    The discriminant of the Booker quartic, :math:`\beta^2 - 4 \alpha \gamma`,
    written as a sum of squares:

    .. math::

        \beta^2 - 4\alpha\gamma = (\epsilon_\perp^2 - \epsilon_g^2 - \epsilon_\parallel \epsilon_\perp)^2 \cos^4\theta_m
                                  + 4 \epsilon_\parallel^2 \epsilon_g^2 \sin^2\theta_m

    (Stix, Waves in Plasmas, eq. 1-27, with sin(theta_m) in place of Stix's
    cos(theta)). This is identical to ``beta**2 - 4*alpha*gamma`` as a
    polynomial, but it is never negative, and it does not cancel near zero
    density, where the two roots nearly coincide. There ``beta**2 - 4*alpha*gamma``
    is a small difference of O(1) numbers, and its square root turns rounding
    into noise of order ``eps * beta**2 / (4 |alpha| sqrt(disc))`` in H, which
    grows like 1/X towards the plasma edge and spoils finite differences of H.

    Parameters
    ----------
    launch_angular_freq : float
        Angular frequency, in rad/s, at launch
    B_total : ArrayLike
        Magnitude of the magnetic field, in Tesla
    sin_theta_m_sq : ArrayLike
        ( sin(theta_m) )^2
    electron_density : ArrayLike
        Electron density, in 10^19 m^-3
    temperature : ArrayLike or None
        Temperature, in keV

    Returns
    ----------
    ArrayLike
        The discriminant, non-negative
    """
    epsilon_para, epsilon_perp, epsilon_g = find_epsilon_terms(
        launch_angular_freq, B_total, electron_density, temperature
    )
    cos_theta_m_sq = 1 - sin_theta_m_sq
    return (epsilon_perp**2 - epsilon_g**2 - epsilon_para * epsilon_perp) ** 2 * (
        cos_theta_m_sq**2
    ) + 4 * epsilon_para**2 * epsilon_g**2 * sin_theta_m_sq


def find_Booker_alpha(
    launch_angular_freq: float,
    B_total: ArrayLike,
    sin_theta_m_sq: ArrayLike,
    electron_density: ArrayLike,
    temperature: Optional[ArrayLike] = None,
) -> ArrayLike:
    """
    Finds `Booker_alpha`

    Parameters
    ----------
    launch_angular_freq : float
        Angular frequency, in rad/s, at launch
    B_total : ArrayLike
        Magnitude of the magnetic field, in Tesla, at the points in the
        plasma to be queried
    sin_theta_m_sq : ArrayLike
        ( sin(theta_m) )^2 at the points in the plasma to be queried
    electron_density : ArrayLike
        Electron density, in 10^19 m^-3, at the points in the plasma to be
        queried
    temperature : ArrayLike or None
        Temperature, in keV, at the points in the plasma to be queried

    Returns
    ----------
    Booker_alpha : ArrayLike
    """
    return find_Booker_terms(
        launch_angular_freq,
        B_total,
        sin_theta_m_sq,
        electron_density,
        temperature,
        alpha=True,
        beta=False,
        gamma=False,
    )[0]


def find_Booker_beta(
    launch_angular_freq: float,
    B_total: ArrayLike,
    sin_theta_m_sq: ArrayLike,
    electron_density: ArrayLike,
    temperature: Optional[ArrayLike] = None,
) -> ArrayLike:
    """
    Finds `Booker_beta`

    Parameters
    ----------
    launch_angular_freq : float
        Angular frequency, in rad/s, at launch
    B_total : ArrayLike
        Magnitude of the magnetic field, in Tesla, at the points in the
        plasma to be queried
    sin_theta_m_sq : ArrayLike
        ( sin(theta_m) )^2 at the points in the plasma to be queried
    electron_density : ArrayLike
        Electron density, in 10^19 m^-3, at the points in the plasma to be
        queried
    temperature : ArrayLike or None
        Temperature, in keV, at the points in the plasma to be queried

    Returns
    ----------
    Booker_beta : ArrayLike
    """
    return find_Booker_terms(
        launch_angular_freq,
        B_total,
        sin_theta_m_sq,
        electron_density,
        temperature,
        alpha=False,
        beta=True,
        gamma=False,
    )[0]


def find_Booker_gamma(
    launch_angular_freq: float,
    B_total: ArrayLike,
    sin_theta_m_sq: ArrayLike,
    electron_density: ArrayLike,
    temperature: Optional[ArrayLike] = None,
) -> ArrayLike:
    """
    Finds `Booker_gamma`

    Parameters
    ----------
    launch_angular_freq : float
        Angular frequency, in rad/s, at launch
    B_total : ArrayLike
        Magnitude of the magnetic field, in Tesla, at the points in the
        plasma to be queried
    sin_theta_m_sq : ArrayLike
        ( sin(theta_m) )^2 at the points in the plasma to be queried
    electron_density : ArrayLike
        Electron density, in 10^19 m^-3, at the points in the plasma to be
        queried
    temperature : ArrayLike or None
        Temperature, in keV, at the points in the plasma to be queried

    Returns
    ----------
    Booker_gamma : ArrayLike
    """
    return find_Booker_terms(
        launch_angular_freq,
        B_total,
        sin_theta_m_sq,
        electron_density,
        temperature,
        alpha=False,
        beta=False,
        gamma=True,
    )[0]


def find_D_terms(
    launch_angular_freq: float,
    K_magnitude: ArrayLike,
    epsilon_para: ArrayLike,
    epsilon_perp: ArrayLike,
    epsilon_g: ArrayLike,
    theta_m: ArrayLike,
) -> List[ArrayLike]:
    """
    Finds `D_11`, `D_22`, `D_bb`, `D_12` and `D_1b` of the cold plasma
    dispersion relation tensor

    Parameters
    ----------
    launch_angular_freq : float
        Angular frequency, in rad/s, at launch
    K_magnitude : ArrayLike
        Magnitude of the wavevector at the points in the plasma to be
        queried
    epsilon_para : ArrayLike
        `epsilon_para` at the points in the plasma to be queried
    epsilon_perp : ArrayLike
        `epsilon_perp` at the points in the plasma to be queried
    epsilon_g : ArrayLike
        `epsilon_g` at the points in the plasma to be queried
    theta_m : ArrayLike
        `theta_m` at the points in the plasma to be queried

    Returns
    ----------
    D_11 : ArrayLike
    D_22 : ArrayLike
    D_bb : ArrayLike
    D_12 : ArrayLike
    D_1b : ArrayLike
    """
    K0 = angular_frequency_to_wavenumber(launch_angular_freq)
    N = K_magnitude / K0
    sin_theta_m = np.sin(theta_m)
    cos_theta_m = np.cos(theta_m)

    D_11 = epsilon_perp - N**2 * sin_theta_m**2
    D_22 = epsilon_perp - N**2
    D_bb = epsilon_para - N**2 * cos_theta_m**2
    D_12 = epsilon_g
    D_1b = N**2 * sin_theta_m * cos_theta_m

    return [D_11, D_22, D_bb, D_12, D_1b]


def find_H_Cardano_eigh(
    launch_angular_freq: float,
    K_magnitude: ArrayLike,
    epsilon_para: ArrayLike,
    epsilon_perp: ArrayLike,
    epsilon_g: ArrayLike,
    theta_m: ArrayLike,
) -> Tuple[FloatArray, ComplexFloatArray]:
    """
    Finds the eigenvectors and eigenvalues of the cold plasma dispersion
    relation tensor with finite mismatch

    Parameters
    ----------
    launch_angular_freq : float
        Angular frequency, in rad/s, at launch
    K_magnitude : ArrayLike
        Magnitude of the wavevector at the points in the plasma to be
        queried
    epsilon_para : ArrayLike
        `epsilon_para` at the points in the plasma to be queried
    epsilon_perp : ArrayLike
        `epsilon_perp` at the points in the plasma to be queried
    epsilon_g : ArrayLike
        `epsilon_g` at the points in the plasma to be queried
    theta_m : ArrayLike
        `theta_m` at the points in the plasma to be queried

    Returns
    ----------
    eigvals : FloatArray
        Eigenvalues of H
    eigvecs : ComplexFloatArray
        Eigenvectors of H, in the same order as eigvals
    """
    D_11, D_22, D_bb, D_12, D_1b = find_D_terms(
        launch_angular_freq, K_magnitude, epsilon_para, epsilon_perp, epsilon_g, theta_m
    )

    if isinstance(K_magnitude, (float, int)):
        length = 1
    else:
        length = K_magnitude.size

    D_tensor = np.zeros([length, 3, 3], dtype=np.complex128)
    D_tensor[:, 0, 0] = D_11
    D_tensor[:, 1, 1] = D_22
    D_tensor[:, 2, 2] = D_bb
    D_tensor[:, 0, 1] = -1j * D_12
    D_tensor[:, 1, 0] = 1j * D_12
    D_tensor[:, 0, 2] = D_tensor[:, 2, 0] = D_1b
    eigvals, eigvecs = np.linalg.eigh(D_tensor)

    return np.squeeze(eigvals), np.squeeze(eigvecs)


def find_H_Cardano_formula(
    launch_angular_freq: float,
    K_magnitude: ArrayLike,
    epsilon_para: ArrayLike,
    epsilon_perp: ArrayLike,
    epsilon_g: ArrayLike,
    theta_m: ArrayLike,
) -> Tuple[FloatArray, FloatArray, FloatArray]:
    r"""
    Finds the Cardano solutions for the cold plasma dispersion
    relation tensor with finite mismatch

    Parameters
    ----------
    launch_angular_freq : float
        Angular frequency, in rad/s, at launch
    K_magnitude : (N,) ArrayLike
        Magnitude of the wavevector at the points in the plasma to be
        queried
    epsilon_para : (N,) ArrayLike
        `epsilon_para` at the points in the plasma to be queried
    epsilon_perp : (N,) ArrayLike
        `epsilon_perp` at the points in the plasma to be queried
    epsilon_g : (N,) ArrayLike
        `epsilon_g` at the points in the plasma to be queried
    theta_m : (N,) ArrayLike
        `theta_m` at the points in the plasma to be queried

    Returns
    ----------
    H_Cardano_1 : (N,) ArrayLike
    H_Cardano_2 : (N,) ArrayLike
    H_Cardano_3 : (N,) ArrayLike
    """
    D_11, D_22, D_bb, D_12, D_1b = find_D_terms(
        launch_angular_freq, K_magnitude, epsilon_para, epsilon_perp, epsilon_g, theta_m
    )

    h_0_coeff = (D_22 * D_1b**2) + (D_bb * D_12**2) - (D_11 * D_22 * D_bb)
    h_1_coeff = (D_11 * D_bb) + (D_11 * D_22) + (D_22 * D_bb) - D_12**2 - D_1b**2
    h_2_coeff = -D_11 - D_22 - D_bb
    h_t_coeff = (
        -2 * h_2_coeff**3
        + 9 * h_2_coeff * h_1_coeff
        - 27 * h_0_coeff
        + 3
        * np.sqrt(3)
        * np.sqrt(
            4 * h_2_coeff**3 * h_0_coeff
            - h_2_coeff**2 * h_1_coeff**2
            - 18 * h_2_coeff * h_1_coeff * h_0_coeff
            + 4 * h_1_coeff**3
            + 27 * h_0_coeff**2
            + 0j  # to make the argument of the np.sqrt complex, so that the sqrt evaluates negative functions
        )
    ) ** (1 / 3)

    H_1_Cardano = (
        h_t_coeff / (3 * 2 ** (1 / 3))
        - 2 ** (1 / 3) * (3 * h_1_coeff - h_2_coeff**2) / (3 * h_t_coeff)
        - h_2_coeff / 3
    )
    H_2_Cardano = (
        -(1 - 1j * np.sqrt(3)) / (6 * 2 ** (1 / 3)) * h_t_coeff
        + (1 + 1j * np.sqrt(3))
        * (3 * h_1_coeff - h_2_coeff**2)
        / (3 * 2 ** (2 / 3) * h_t_coeff)
        - h_2_coeff / 3
    )
    H_3_Cardano = (
        -(1 + 1j * np.sqrt(3)) / (6 * 2 ** (1 / 3)) * h_t_coeff
        + (1 - 1j * np.sqrt(3))
        * (3 * h_1_coeff - h_2_coeff**2)
        / (3 * 2 ** (2 / 3) * h_t_coeff)
        - h_2_coeff / 3
    )
    return H_1_Cardano, H_2_Cardano, H_3_Cardano


def find_mode_index_and_ehat(
    mode_flag: Literal[1, -1, "O", "X"],
    H_Cardanos: FloatArray,
    ehats: ComplexFloatArray,
    tol_H: float = 1e-5,
    tol_O_mode_polarisation: float = 0.25,
) -> Tuple[int, float, ComplexFloatArray]:
    """
    Given an array `H_Cardanos` of shape (3,) and an array of `ehats` of
    shape (3,3) where `H_Cardanos[i]` corresponds to `ehats[:,i]`, choose
    the element of `H_Cardanos` which has the smallest absolute value; i.e.
    choose `idx` such that `np.min(H_Cardanos) == abs(H_Cardanos[idx])`.
    This corresponds to the solution to be used to calculate the correct
    mode solution during post-processing

    Parameters
    ----------
    mode_flag : VALID_LAUNCH_MODE_FLAGS
        Must be one of [1, -1, "O", "X"]
    H_Cardanos : FloatArray
        Array of floats corresponding to the eigenvalues of the cold
        plasma dispersion relation tensor
    ehats : ComplexFloatArray
        Array of vectors corresponding to the eigenvectors of the cold
        plasma dispersion relation tensor, in the same order as
        `H_Cardanos`
    tol_H : float
        Absolute tolerance for the solution in satisfying H=0
    tol_O_mode_polarisation : float
        Absolute tolerance to check if the polarisation vector corresponds
        to O- or X-mode; i.e. if `e_hat[-1] < tol_O_mode_polarisation` then
        it is X-mode, otherwise it's O-mode

    Returns
    ----------
    mode_index : int
        Index of the solution corresponding to the desired mode_flag
    H : float
        Value of the solution; i.e. `H_Cardanos[mode_index]`
    ehat : ComplexFloatArray
        Polarisation vector of the chosen solution; i.e. `ehats[mode_index]`
    """
    soln_idxs = np.where(np.abs(H_Cardanos) <= tol_H)[0]
    soln_H = H_Cardanos[soln_idxs]
    soln_ehats = ehats[
        :, soln_idxs
    ]  # note that soln_H[i] still corresponds to soln_ehats[:,i]

    # We have three cases to check for:
    #   (i)   If there are none or three solutions, raise an error
    #   (ii)  If there is one solution, just return that
    #   (iii) If there is more than 1 solution and the user specifies `mode_flag` = 1 or -1,
    # then we have no way of knowing what exactly the user wants (i.e. we don't know
    # if the O-mode or X-mode solution is wanted), so raise an error
    #         Otherwise, check which mode index yields the desired mode

    if soln_idxs.shape in [(0,), (3,)]:
        raise RuntimeError(
            f"Unable to check which mode index corresponds to O- and X-mode. Found {soln_idxs.shape} solutions!"
        )

    elif soln_idxs.shape == (1,):
        mode_idx = soln_idxs[0]
        H = soln_H[0]
        ehat = soln_ehats

    elif soln_idxs.shape == (2,) and mode_flag in [1, -1]:
        raise RuntimeError(
            f"Specific mode to be used not specified (`mode_flag` = {mode_flag}) and unable to determine which mode indices correspond to O- and X-mode"
        )

    else:  # soln_idxs.shape == (2,) and mode_flag in ["O", "X"]:
        H0, H1 = H_Cardanos[soln_idxs]
        e0, e1 = soln_ehats.T  # because soln_ehats.shape == (3,2)
        abs_Re_e0_b = abs(np.real(e0[-1]))
        abs_Re_e1_b = abs(np.real(e1[-1]))

        if (
            abs_Re_e0_b < 1 - tol_O_mode_polarisation
            and abs_Re_e1_b < 1 - tol_O_mode_polarisation
        ):
            log.warning(
                f"Both computed vectors ({e0} and {e1}) are not within the O-mode polarisation tolerance (tol = {tol_O_mode_polarisation})"
            )
            log.warning(
                f"Setting the vector with the largets b-component to be the O-mode. This may or may not cause issues"
            )

        if mode_flag == "O":
            idx = 0 if abs_Re_e0_b > abs_Re_e1_b else 1
        else:
            idx = 1 if abs_Re_e0_b > abs_Re_e1_b else 0

        mode_idx = soln_idxs[idx]
        H = soln_H[idx]
        ehat = soln_ehats[idx]

    return np.squeeze(mode_idx), np.squeeze(H), np.squeeze(ehat)


def find_beam_widths_and_curvs(
    Psi_w: ComplexFloatArray, K_vec: FloatArray, g_hat: FloatArray
):
    """
    Calculate the beam widths and curvatures

    Parameters
    ----------
    Psi_w : (2,2) or (N,2,2) ComplexFloatArray
        The beam matrix
    K_vec : (3,) or (N,3) FloatArray
        The wavevector
    g_hat : (3,) or (N,3) FloatArray
        The group velocity unit vector

    Returns
    ----------
    curv1 : float or (N,) FloatArray
    curv2 : float or (N,) FloatArray
    width1 : float or (N,) FloatArray
    width2 : float or (N,) FloatArray
    """

    Re_Psi_w = np.real(Psi_w)
    Re_Psi_w_eigvals, Re_Psi_w_eigvecs = np.linalg.eigh(Re_Psi_w)  # (N,2) and (N,2,2)
    Re_Psi_w_eigvals = check_eigval_order(Re_Psi_w_eigvals, Re_Psi_w_eigvecs)  # type: ignore
    K_mag = np.linalg.norm(K_vec, axis=1)
    K_g_mag = np.sum(K_vec * g_hat, axis=1)
    curvs = np.squeeze((K_g_mag**2 / K_mag**3)[:, np.newaxis] * Re_Psi_w_eigvals)

    Im_Psi_w = np.imag(Psi_w)
    Im_Psi_w_eigvals, Im_Psi_w_eigvecs = np.linalg.eigh(Im_Psi_w)
    Im_Psi_w_eigvals = check_eigval_order(Im_Psi_w_eigvals, Im_Psi_w_eigvecs)  # type: ignore
    widths = np.squeeze(np.sqrt(np.full(Im_Psi_w_eigvals.shape, 2) / Im_Psi_w_eigvals))

    if curvs.ndim == 1:
        return curvs[0], curvs[1], widths[0], widths[1]
    else:
        return curvs[:, 0], curvs[:, 1], widths[:, 0], widths[:, 1]


def check_eigval_order(
    eigvals: FloatArray, eigvecs: FloatArray, diff_tol: float = 0.1
) -> FloatArray:
    """
    Check if the eigenvalues are correctly ordered by comparing
    each eigenvector of `eigvec1`

    Parameters
    ----------
    eigvals : (N,2) FloatArray
    eigvecs : (N,2,2) FloatArray
        Note that the eigenvector `e_vec[:,:,i]` is the eigenvector
        of the i-th eigenvalue `e_val[:,i]`

    Returns
    ----------
    eigvals_sorted : (N,2) FloatArray
        Correctly ordered array of eigenvalues
    """
    eigval1 = eigvals[:, 0]
    eigval2 = eigvals[:, 1]
    eigvec1 = eigvecs[:, :, 0].T  # (2, N)
    arr1 = eigvec1[:-1]  # (2, N-1)
    arr2 = eigvec1[1:]  # (2, N-1)
    diff = np.linalg.norm(arr1 - arr2, axis=0)  # (N-1,)
    diff_mask = np.where(diff < diff_tol)[0]
    if diff_mask.shape == (0,):
        for i in diff_mask:
            idx = i + 1
            _temp = eigval1[idx:]
            eigval1[idx:] = eigval2[idx:]
            eigval2[idx:] = _temp

    return np.array([eigval1, eigval2]).T  # (N, 2)


##################################################
#
# GENERAL FUNCTIONS
#
##################################################


def read_floats_into_list_until(terminator: str, lines: TextIO) -> FloatArray:
    """Reads the lines of a file until the string (terminator) is read.

    Currently used to read topfile.

    Written by NE Bricknell
    """
    lst: List[float] = []
    while True:
        try:
            line = lines.readline()
        except StopIteration:
            break
        if terminator in line:
            break
        elif not line:
            break
        lst.extend(map(float, line.split()))
    return np.asarray(lst)


def find_nearest(array: ArrayLike, value: Any) -> int:
    """Returns the index of the first element in ``array`` closest in
    absolute value to ``value``

    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return int(idx)


def ray_line(
    q_X_launch: float,
    q_Y_launch: float,
    q_Z_launch: float,
    tau: Union[float, int, FloatArray],
    poloidal_launch_angle_deg_Torbeam: float,
    toroidal_launch_angle_deg_Torbeam: float,
) -> FloatArray:
    """
    Calculates the position vectors of a straight-line trajectory
    parametrised by `tau` from a launch position `(q_X_launch, q_Y_launch,
    q_Z_launch)` with launch angles `poloidal_launch_angle_deg_Torbeam`
    and `toroidal_launch_angle_deg_Torbeam`

    Parameters
    ----------
    q_X_launch, q_Y_launch, q_Z_launch : float
        Position of the launch position
    tau : float, int, FloatArray
        Parameter of the ray trajectory
    poloidal_launch_angle_deg_Torbeam : float
        Poloidal launch angle, in the TORBEAM convention, in degrees
    toroidal_launch_angle_deg_Torbeam : float
        Toroidal launch angle, in the TORBEAM convention, in degrees

    Returns
    ----------
    output : (3,) or (3,N) FloatArray
        Coordinates, in Cartesian, of the ray trajectory at points `tau`,
        with the same length as `tau`
    """

    XYZ_start = np.array([q_X_launch, q_Y_launch, q_Z_launch])

    # This parametrises the ray in a line away from the antenna,
    # at a/up to some values of the parameter `tau`
    poloidal_launch_angle = -np.deg2rad(poloidal_launch_angle_deg_Torbeam)
    toroidal_launch_angle = np.deg2rad(toroidal_launch_angle_deg_Torbeam) + np.pi
    XYZ_step = np.array(
        [
            np.cos(toroidal_launch_angle) * np.cos(poloidal_launch_angle),
            np.sin(toroidal_launch_angle) * np.cos(poloidal_launch_angle),
            np.sin(poloidal_launch_angle),
        ]
    )

    ray_line_positions = XYZ_start + np.outer(tau, XYZ_step)

    return np.squeeze(ray_line_positions).T


def poloidal_flux_along_ray_line(
    q_X_launch: float,
    q_Y_launch: float,
    q_Z_launch: float,
    tau: Union[float, int, FloatArray],
    poloidal_launch_angle_deg_Torbeam: float,
    toroidal_launch_angle_deg_Torbeam: float,
    _field_cart_polflux: Callable,
) -> FloatArray:
    """
    Calculates the poloidal flux values along a straight-line trajectory
    parametrised by `tau` from a launch position `(q_X_launch, q_Y_launch,
    q_Z_launch)` with launch angles `poloidal_launch_angle_deg_Torbeam`
    and `toroidal_launch_angle_deg_Torbeam`

    Parameters
    ----------
    q_X_launch, q_Y_launch, q_Z_launch : float
        Position of the launch position
    tau : float, int, FloatArray
        Parameter of the ray trajectory
    poloidal_launch_angle_deg_Torbeam : float
        Poloidal launch angle, in the TORBEAM convention, in degrees
    toroidal_launch_angle_deg_Torbeam : float
        Toroidal launch angle, in the TORBEAM convention, in degrees
    _field_cart_polflux : Callable
        Function which takes in Cartesian coordinates and returns the
        poloidal flux at those points

    Returns
    ----------
    output : (3,) or (3,N) FloatArray
        Poloidal flux values along the ray trajectory at points `tau`,
        with the same length as `tau`
    """

    positions = ray_line(
        q_X_launch,
        q_Y_launch,
        q_Z_launch,
        tau,
        poloidal_launch_angle_deg_Torbeam,
        toroidal_launch_angle_deg_Torbeam,
    )

    # Get the poloidal flux values at a particular point or point(s).
    # If there are NaNs, then replace those with the first non-NaN instance
    # in the array (i.e. the poloidal flux of the first point in the field)
    if positions.ndim == 1:
        polflux = _field_cart_polflux(*positions)
        if np.isnan(polflux):
            log.warning(
                f"Poloidal flux is NaN at [X,Y,Z] = [{arr2str(positions)}]. Returning None instead"
            )
            polflux = np.nan

    elif positions.ndim == 2:
        polflux = _field_cart_polflux(positions[:, 0], positions[:, 1], positions[:, 2])
        if np.all(np.isnan(polflux)):
            _printmsg = "\n".join(
                f"            - [{position}]" for position in positions
            )
            log.warning(f"""
        Poloidal fluxes are NaNs at all points queried: [X,Y,Z] = \n{_printmsg}
        """)
            polflux = np.nan
        else:
            NaN_replacement_value = polflux[~np.isnan(polflux)][0]
            polflux = np.nan_to_num(polflux, nan=NaN_replacement_value)

    else:
        raise ValueError(
            f"Ray line positions must have only 1 or 2 dimensions but got {positions.ndim}"
        )

    return polflux  # type: ignore


def poloidal_flux_difference_along_ray_line(
    q_X_launch: float,
    q_Y_launch: float,
    q_Z_launch: float,
    tau: Union[float, int, FloatArray],
    poloidal_launch_angle_deg_Torbeam: float,
    toroidal_launch_angle_deg_Torbeam: float,
    _field_cart_polflux: Callable,
    poloidal_flux_enter: float,
) -> FloatArray:
    """
    Calculates the poloidal flux difference, relative to the
    `poloidal_flux_enter`, along a straight-line trajectory
    parametrised by `tau` from a launch position `(q_X_launch, q_Y_launch,
    q_Z_launch)` with launch angles `poloidal_launch_angle_deg_Torbeam`
    and `toroidal_launch_angle_deg_Torbeam`

    Parameters
    ----------
    q_X_launch, q_Y_launch, q_Z_launch : float
        Position of the launch position
    tau : float, int, FloatArray
        Parameter of the ray trajectory
    poloidal_launch_angle_deg_Torbeam : float
        Poloidal launch angle, in the TORBEAM convention, in degrees
    toroidal_launch_angle_deg_Torbeam : float
        Toroidal launch angle, in the TORBEAM convention, in degrees
    _field_cart_polflux : Callable
        Function which takes in Cartesian coordinates and returns the
        poloidal flux at those points
    poloidal_flux_enter : float
        Poloidal flux value to begin the ray-/beam-tracing calculations at

    Returns
    ----------
    output : (3,) or (3,N) FloatArray
        Poloidal flux difference, relative to `poloidal_flux_enter` along
        the ray trajectory at points `tau`, with the same length as `tau`
    """

    polflux = poloidal_flux_along_ray_line(
        q_X_launch,
        q_Y_launch,
        q_Z_launch,
        tau,
        poloidal_launch_angle_deg_Torbeam,
        toroidal_launch_angle_deg_Torbeam,
        _field_cart_polflux,
    )

    # If NaN, then just return NaN
    if polflux is np.nan:
        return polflux
    else:
        return polflux - poloidal_flux_enter


# TODO -- rewrite using Einstein summation convention?
def dot(arr_a: Array, arr_b: Array) -> Array:
    """
    Dot product of arrays of vectors or matrices, which covers the case
    that `matmul` and `dot` don't do very elegantly, and avoids having to
    use a for loop to iterate over the array slices. Replaces the
    deprecated `contract_special`

    Parameters
    ----------
    arg_a, arg_b:
        One of:
        - matrix with shape TxMxN and a vector with TxN or TxM
        - two vectors of shape TxN

        For each T, independently compute the dot product of the
        matrices/vectors
    """
    return np.array(list(map(np.dot, arr_a, arr_b)))


def make_array_3x3(arr: Array) -> Array:
    r"""
    Convert a 2x2 array into a 3x3 by appending zeros on the outside:

    .. math::

        \begin{pmatrix}
            a & b \\
            c & d
        \end{pmatrix}
        \Rightarrow
        \begin{pmatrix}
            a & b & 0 \\
            c & d & 0 \\
            0 & 0 & 0
        \end{pmatrix}
    
    Parameters
    ----------
    arr : Array
        The input 2x2 array to expand
    
    Returns
    ----------
    output : Array
        A 3x3 array with zeroes appended to the right and bottom
    """
    if arr.shape != (2, 2):
        raise ValueError(f"Expected array shape to be (2, 2), got {arr.shape}")
    out = np.zeros((3, 3), dtype=arr.dtype)
    out[:2, :2] = arr
    return out


def make_unit_vector_from_cross_product(
    vector_a: FloatArray, vector_b: FloatArray
) -> FloatArray:
    """
    Take the cross product of two vectors and then normalise

    Old documentation:
        Assume np.shape(vector_a) = np.shape(vector_b) = (n,3)
        or
        np.shape(vector_a) = (n,3) , np.shape(vector_b) = (3)

    Parameters
    ----------
    vector_a : (3,) or (N,3) FloatArray
    vector_b : (3,) or (N,3) FloatArray

    Returns
    ----------
    output_unit_vector : (3,) or (N,3) FloatArray
    """
    output_vector = np.cross(vector_a, vector_b)
    output_vector_magnitude = np.linalg.norm(output_vector, axis=-1)
    output_unit_vector = output_vector / np.tile(output_vector_magnitude, (3, 1)).T

    return output_unit_vector


def find_inverse_2D(matrix_2D: Array) -> Array:
    """
    Find the inverse M^-1 of a 2x2 matrix M; i.e. M x M^-1 = I. If matrix
    M is close to singular, then raise an error; i.e. when | det(M) | < 1e-5

    Parameters
    ----------
    matrix_2D : (2,2) Array
        The input 2x2 array to invert

    Returns
    ----------
    matrix_2D_inverse : (2,2) Array
        The inverse array
    """
    # Finds the inverse of a 2x2 matrix
    matrix_2D_inverse = np.zeros([2, 2], dtype=matrix_2D.dtype)
    det = matrix_2D[0, 0] * matrix_2D[1, 1] - matrix_2D[0, 1] * matrix_2D[1, 0]
    if np.abs(det) < 1e-5:
        log.warning(f"Matrix is close to singular")
    matrix_2D_inverse[0, 0] = matrix_2D[1, 1] / det
    matrix_2D_inverse[1, 1] = matrix_2D[0, 0] / det
    matrix_2D_inverse[0, 1] = -matrix_2D[0, 1] / det
    matrix_2D_inverse[1, 0] = -matrix_2D[1, 0] / det
    return matrix_2D_inverse


def find_g_cartesian(
    cartesian: bool, q0: ArrayLike, q1: ArrayLike, q2: ArrayLike, dH: dict
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Find, in Cartesian coordinates:
    (1) the group velocity vector, `g`;
    (2) its magnitude, `|g|`; and
    (3) its unit vector, `g_hat`.

    Parameters
    ----------
    cartesian : bool
        `True` for Cartesian geometry and `False` for cylindrical
    q0 : ArrayLike
        `q_X` for Cartesian geometry and `q_R` for cylindrical
    q1 : ArrayLike
        `q_Y` for Cartesian geometry and `q_zeta` for cylindrical
    q2 : ArrayLike
        `q_Z` for both Cartesian and cylindrical geometry
    K0 : ArrayLike
        `K_X` for Cartesian geometry and `K_R` for cylindrical
    K1 : ArrayLike
        `K_Y` for Cartesian geometry and `K_zeta` for cylindrical
    K2 : ArrayLike
        `K_Z` for both Cartesian and cylindrical geometry
    dH : dict
        Dictionary of derivatives of the Hamiltonian, i.e. `dH = hamiltonian.derivatives(...)`

    Returns
    ----------
    g_vec : ArrayLike
        Group velocity, in Cartesian coordinates
    g_mag : ArrayLike
        Magnitude of the group velocity
    g_hat : ArrayLike
        Unit vector of the group velocity, in Cartesian coordinates
    """
    if cartesian:
        dH_dKx = dH["dH_dKX"]
        dH_dKy = dH["dH_dKY"]
        dH_dKz = dH["dH_dKZ"]
    else:  # cylindrical
        q_R, q_zeta, q_Z = q0, q1, q2
        sin_zeta = np.sin(q_zeta)
        cos_zeta = np.cos(q_zeta)
        dH_dKR = dH["dH_dKR"]
        dH_dKzeta = dH["dH_dKzeta"]
        dH_dKx = cos_zeta * dH_dKR - q_R * sin_zeta * dH_dKzeta
        dH_dKy = sin_zeta * dH_dKR + q_R * cos_zeta * dH_dKzeta
        dH_dKz = dH["dH_dKZ"]

    g_vec_cart = np.stack((dH_dKx, dH_dKy, dH_dKz), axis=1)
    g_magnitude = np.linalg.norm(g_vec_cart, axis=1)
    g_hat_cart = g_vec_cart / g_magnitude[:, np.newaxis]

    return g_vec_cart, g_magnitude, g_hat_cart


def find_K_magnitude(
    cartesian: bool, K0: ArrayLike, K1: ArrayLike, K2: ArrayLike, q0: ArrayLike
) -> ArrayLike:
    """
    Find the magnitude of the wavevector. If in cylindrical geometry (i.e.
    `K_lab_cyl`) then account for the fact that `K_lab_cyl[1]` is the
    toroidal mode number

    Parameters
    ----------
    cartesian : bool
        `True` for Cartesian geometry and `False` for cylindrical
    K0 : ArrayLike
        `K_X` for Cartesian geometry and `K_R` for cylindrical
    K1 : ArrayLike
        `K_Y` for Cartesian geometry and `K_zeta` for cylindrical
    K2 : ArrayLike
        `K_Z` for both Cartesian and cylindrical geometry
    q0 : ArrayLike
        `q_R` for cylindrical geometry; unused if Cartesian

    Returns
    ----------
    output : ArrayLike
        Magnitude of the wavevector
    """
    return (
        np.sqrt(K0**2 + K1**2 + K2**2)
        if cartesian
        else np.sqrt(K0**2 + (K1 / q0) ** 2 + K2**2)
    )


def reshape_33N_to_N33(array: FloatArray):
    """Such that shape is [points,3,3] instead of [3,3,points]"""
    if array.ndim == 2:
        return array
    return np.moveaxis(np.squeeze(array), 2, 0)


##################################################
#
# GAUSSIAN BEAM FUNCTIONS
#
##################################################


def find_Rayleigh_length(waist: float, wavenumber: float):
    """
    Finds the Rayleigh length (assumes vacuum propagation and circular beam)

    Parameters
    ----------
    waist : float
    wavenumber : float

    Returns
    ----------
    rayleigh_length : float
    """
    return 0.5 * wavenumber * waist**2


def find_waist(width: float, wavenumber: float, curvature: float):
    """
    Finds the size of the waist assuming vacuum propagation and circular beam

    Parameters
    ----------
    width : float
    wavenumber : float
    curvature : float

    Returns
    ----------
    beam_waist : float
    """
    return width / np.sqrt(1 + curvature**2 * width**4 * wavenumber**2 / 4)


def find_distance_from_waist(width: float, wavenumber: float, curvature: float):
    """
    Finds how far (signed distance) you are from the waist (assumes vacuum propagation and circular beam)

    Parameters
    ----------
    width : float
    wavenumber : float
    curvature : float

    Returns
    ----------
    distance_from_waist : float
        Signed distance from the waist; negative for distance to, positive for distance
        after waist
    """
    waist_sq = find_waist(width, wavenumber, curvature) ** 2
    return np.sign(curvature) * np.sqrt(
        (width**2 - waist_sq) * waist_sq * wavenumber**2 / 4
    )


def propagate_beam(
    Psi_w_cartesian: ComplexFloatArray, propagation_distance: float, freq_GHz: float
) -> ComplexFloatArray:
    """
    Uses the vacuum solution of the beam tracing equations to propagate the beam
    (works for arbitrary Gaussian beam in vacuum)

    Parameters
    ----------
    Psi_w_cartesian : (2,2) ComplexFloatArray
    propagation_distance : float
    freq_GHz : float

    Returns
    ----------
    Psi_w_cartesian_output : (2,2) ComplexFloatArray
    """
    wavenumber_K0 = freq_GHz_to_wavenumber(freq_GHz)

    Psi_w_inv_cartesian = find_inverse_2D(Psi_w_cartesian)

    Psi_w_inv_final_cartesian = (
        propagation_distance / (wavenumber_K0) * np.eye(2) + Psi_w_inv_cartesian
    )

    Psi_w_final_cartesian = find_inverse_2D(Psi_w_inv_final_cartesian)

    return cast(ComplexFloatArray, Psi_w_final_cartesian)


def propagate_circular_beam(
    width: float, curvature: float, propagation_distance: float, freq_GHz: float
) -> Tuple[float, float]:
    """
    Directly uses the width and curvature formulae to calculate circular
    beam propagation (assumes vacuum propagation and circular beam)

    Parameters
    ----------
    width : float
    curvature : float
    propagation_distance : float
    freq_GHZ : float

    Returns
    ----------
    width : float
    curvature : float
    """
    wavenumber = freq_GHz_to_wavenumber(freq_GHz)

    w0 = find_waist(width, wavenumber, curvature)
    z0 = find_distance_from_waist(width, wavenumber, curvature)
    z_R = find_Rayleigh_length(w0, wavenumber)

    z = z0 + propagation_distance

    widths = w0 * np.sqrt(1 + (z / z_R) ** 2)
    curvatures = z / (z**2 + z_R**2)
    return widths, curvatures


##################################################
#
# LAUNCH ANGLES FUNCTIONS
#
##################################################


def use_deg_args(func, *args, **kwargs):
    "Transform args from degrees to radians before calling"
    return func(*[a * np.pi / 180 for a in args], **kwargs)


def tilt_trns_RZ_make(t: float) -> FloatArray:
    """
    Tilts R ([1,0,0]) in to Z ([0,0,1]) using angle in radians

    Parameters
    ----------
    t : float
        Tilt angle in radians

    Returns
    ----------
    rot_matrix : (3,3) FloatArray
        Rotation matrix in RZ corresponding to a tilt of `t` radians
    """
    ctilt = np.cos(t)
    stilt = np.sin(t)
    return np.array([[ctilt, 0, stilt], [0, 1, 0], [-stilt, 0, ctilt]])


def tilt_trns_TZ_make(t: float) -> FloatArray:
    """
    Tilts T ([0,1,0]) in to Z ([0,0,1]) using angle in radians

    Parameters
    ----------
    t : float
        Tilt angle in radians

    Returns
    ----------
    rot_matrix : (3,3) FloatArray
        Rotation matrix in TZ corresponding to a tilt of `t` radians
    """
    ctilt = np.cos(t)
    stilt = np.sin(t)
    return np.array([[1, 0, 0], [0, ctilt, -stilt], [0, stilt, ctilt]])


def rot_trns_make(r: float) -> FloatArray:
    """
    Rotates R ([1,0,0]) into T ([0,1,0]) using angle in radians

    Parameters
    ----------
    r : float
        Rotation angle in radians

    Returns
    ----------
    rot_matrix : (3,3) FloatArray
        Rotation matrix in RT corresponding to a rotation of `r` radians
    """
    crot = np.cos(r)
    srot = np.sin(r)
    return np.array([[crot, -srot, 0], [srot, crot, 0], [0, 0, 1]])


def mirrornorm_make_with_rot_tilt_operators(r: float, t: float) -> FloatArray:
    """
    Makes norm vector for mirror using tilt and rot operators (in that order) given
    angles in radians

    Parameters
    ----------
    r : float
        Rotation angle in radians
    t : float
        Tilt angle in radians

    Returns
    ----------
    mirrornorm : (3,) FloatArray
        Mirror normal vector
    """
    mirrornorm0 = np.array([0, 1, 0])
    tilt_trns = tilt_trns_TZ_make(t)
    rot_trns = rot_trns_make(np.pi / 4 + r)
    mirrornorm = rot_trns @ (tilt_trns @ mirrornorm0)
    return mirrornorm


def mirrornorm_make(r: float, t: float) -> FloatArray:
    """
    Makes norm vector for mirror directly using angles in radians

    Parameters
    ----------
    r : float
        Rotation angle in radians
    t : float
        Tilt angle in radians

    Returns
    ----------
    mirror_norm : (3,) FloatArray
        Mirror normal vector
    """
    return np.array(
        [
            *(np.cos(t) * np.array([-np.sin(r + np.pi / 4), np.cos(r + np.pi / 4)])),
            np.sin(t),
        ]
    )


def reflector_make_from_mirrornorm(mirrornorm: FloatArray) -> FloatArray:
    """
    Makes Householder (reflector) matrix for mirror using norm vector for mirror

    Parameters
    ----------
    mirror_norm : (3,) FloatArray
        Mirror normal vector

    Returns
    ----------
    reflector : (3,3) FloatArray
        Reflector matrix
    """
    reflector = np.eye(3) - 2 * mirrornorm[:, np.newaxis] @ mirrornorm[np.newaxis, :]
    return reflector


def reflector_make(r: float, t: float) -> FloatArray:
    """
    Makes Householder (reflector) matrix for mirror using angles in radians

    Parameters
    ----------
    r : float
        Rotation angle in radians
    t : float
        Tilt angle in radians

    Returns
    ----------
    reflector : (3,3) FloatArray
        Reflector matrix
    """
    mirrornorm = mirrornorm_make(r, t)
    reflector = reflector_make_from_mirrornorm(mirrornorm)
    return reflector


def genray_angles_from_mirror_angles(
    rot_ang_deg: ArrayLike,
    tilt_ang_deg: ArrayLike,
    offset_for_window_norm_to_R: ArrayLike = 3.0,
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Compute the GENRAY-equivalent angles from mirror angles

    ``rot_ang_deg = 0``, ``tilt_ang_deg = 0`` implies beam propagates
    in negative window normal direction.

    Can get from ``rot_ang_deg, tilt_ang_deg`` by using toroidal and
    poloidal "genray angles" (``tor_ang, pol_ang``) in IDL savefile
    log. The angles were calculated according to the logic::

        tor_ang = 3.0 - rot_ang_deg
        pol_ang = tilt_ang_deg


    Parameters
    ----------
    rot_ang_deg : ArrayLike
        Steering mirror rotation angle in degrees
    tilt_ang_deg : ArrayLike
        Steering mirror tilt angle in degrees
    offset_for_window_norm_to_R : ArrayLike, default = 3.0
        Toroidal angle in degrees of window normal to major radial
        position, such that for ``rot_ang_deg = 0``, ``tilt_ang_deg =
        0`` beam would have toroidal angle given by
        ``offset_for_window_norm_to_R``.
    """

    beam0 = np.array([0, -1, 0])
    rdeg = rot_ang_deg if np.isscalar(rot_ang_deg) else rot_ang_deg.flat[0]
    tdeg = tilt_ang_deg if np.isscalar(tilt_ang_deg) else tilt_ang_deg.flat[0]
    reflector = use_deg_args(reflector_make, rdeg, tdeg)
    beam_windowframe = reflector @ beam0
    # rotate from window frame to port frame:
    beam = use_deg_args(rot_trns_make, -offset_for_window_norm_to_R) @ beam_windowframe

    tor_ang_genray = np.arctan2(
        beam[1], -beam[0]
    )  # +offset_for_window_norm_to_R*np.pi/180
    pol_ang_genray = np.arctan2(beam[2], np.sqrt(beam[0] ** 2 + beam[1] ** 2))
    tor_ang_genray_deg, pol_ang_genray_deg = (
        tor_ang_genray * 180 / np.pi,
        pol_ang_genray * 180 / np.pi,
    )

    return tor_ang_genray_deg, pol_ang_genray_deg
