# Copyright 2023 - 2023, Valerian Hall-Chen and the Scotty contributors
# SPDX-License-Identifier: GPL-3.0

from scotty.cart_geometry import CartMagneticField
from scotty.profile_fit import ProfileFitLike
from scotty.derivatives import derivative
from scotty.fun_general import (
    angular_frequency_to_wavenumber,
    find_normalised_gyro_freq,
    find_normalised_plasma_freq,
    dot,
)
from scotty.typing import ArrayLike, FloatArray

import numpy as np
from typing import Dict, Tuple, Optional


class DielectricTensor:
    r"""Calculates the components of the cold plasma dielectric tensor for a
    wave with angular frequency :math:`\Omega`:

    .. math::

        \epsilon =
          \begin{pmatrix}
            \epsilon_{11}  & -i\epsilon_{12} & 0 \\
            i\epsilon_{12} & \epsilon_{11}   & 0 \\
            0            & 0             & \epsilon_{bb} \\
          \end{pmatrix}

    where:

    .. math::

        \begin{align}
          \epsilon_{11} &= 1 - \frac{\Omega_{pe}^2}{\Omega^2 - \Omega_{ce}^2} \\
          \epsilon_{12} &=
            1 - \frac{\Omega_{pe}^2\Omega_{ce}}{\Omega(\Omega^2 - \Omega_{ce}^2)} \\
          \epsilon_{bb} &= 1 - \frac{\Omega_{pe}^2}{\Omega^2} \\
        \end{align}

    The components of the dielectric tensor are calculated in the
    :math:`(\hat{\mathbf{u}}_1, \hat{\mathbf{u}}_2, \hat{\mathbf{b}})` basis.
    Hence, :math:`\epsilon_{11}`, :math:`\epsilon_{12}`, and
    :math:`\epsilon_{bb}` correspond to the ``S``, ``D``, and ``P`` variables in
    Stix, respectively. The notation used in this code is chosen to be
    consistent with Hall-Chen, Parra, Hillesheim, PPCF 2022.

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
        temperature: Optional[ArrayLike] = None,
    ):
        _plasma_freq_2 = (
            find_normalised_plasma_freq(
                electron_density, angular_frequency, temperature
            )
            ** 2
        )
        _gyro_freq = find_normalised_gyro_freq(B_total, angular_frequency, temperature)
        _gyro_freq_2 = _gyro_freq**2

        self._epsilon_bb = 1 - _plasma_freq_2
        self._epsilon_11 = 1 - _plasma_freq_2 / (1 - _gyro_freq_2)
        self._epsilon_12 = _plasma_freq_2 * _gyro_freq / (1 - _gyro_freq_2)

    @property
    def e_bb(self):
        r"""The :math:`\epsilon_{bb}` component"""
        return self._epsilon_bb

    @property
    def e_11(self):
        r"""The :math:`\epsilon_{11}` component"""
        return self._epsilon_11

    @property
    def e_12(self):
        r"""The :math:`\epsilon_{12}` component"""
        return self._epsilon_12


class cart_Hamiltonian:
    r"""Functor to evaluate derivatives of the Hamiltonian, :math:`H`, at a
    given set of points.

    ``Scotty`` calculates derivatives using a grid-free finite difference
    approach. The Hamiltonian is evaluated at, essentially, an arbitrary set of
    points around the location we wish to get the derivatives at. In practice we
    define stencils as relative offsets from a central point, and the evaluation
    points are the product of the spacing in a given direction with the stencil
    offsets. By carefully choosing our stencils and evaluating all of the
    derivatives at once, we can reuse evaluations of :math:`H` between
    derivatives, saving a lot of computation.

    The stencils are defined as a `dict` with a `tuple` of offsets as keys and
    `float` weights as values. For example, the `CFD1_stencil`::

        {(1,): 0.5, (-1,): -0.5}

    defines the second-order first central-difference:

    .. math::

        f' = \frac{f(x + \delta_x) - f(x - \delta_x)}{2\delta_x}

    The keys are tuples so that we can iterate over the offsets for the mixed
    second derivatives.

    The stencils have been chosen to maximise the reuse of Hamiltonian
    evaluations without sacrificing accuracy.

    Parameters
    ----------
    field
        An object describing the magnetic field of the plasma
    launch_angular_frequency
        Angular frequency of the beam
    mode_flag
        Either ``+/-1``, used to determine which mode branch to use
    density_fit
        Function or ``Callable`` parameterising the density
    delta_R
        Finite difference spacing in the ``R`` direction
    delta_Z
        Finite difference spacing in the ``Z`` direction
    delta_K_R
        Finite difference spacing in the ``K_R`` direction
    delta_K_zeta
        Finite difference spacing in the ``K_zeta`` direction
    delta_K_Z
        Finite difference spacing in the ``K_Z`` direction

    """

    def __init__(
        self,
        field: CartMagneticField,
        launch_angular_frequency: float,
        mode_flag: int,
        density_fit: ProfileFitLike,
        delta_X: float,
        delta_Y: float,
        delta_Z: float,
        delta_K_X: float,
        delta_K_Y: float,
        delta_K_Z: float,
        temperature_fit: Optional[ProfileFitLike] = None,
    ):
        self.field = field
        self.angular_frequency = launch_angular_frequency
        self.wavenumber_K0 = angular_frequency_to_wavenumber(launch_angular_frequency)
        self.mode_flag = mode_flag
        self.density = density_fit
        self.temperature = temperature_fit
        self.spacings = {
            "q_X": delta_X,
            "q_Y": delta_Y,
            "q_Z": delta_Z,
            "K_X": delta_K_X,
            "K_Y": delta_K_Y,
            "K_Z": delta_K_Z,
        }

    def __call__(
        self,
        q_X: ArrayLike,
        q_Y: ArrayLike,
        q_Z: ArrayLike,
        K_X: ArrayLike,
        K_Y: ArrayLike,
        K_Z: ArrayLike,
    ):
        """Evaluate the Hamiltonian at the given coordinates

        Parameters
        ----------
        q_R : ArrayLike
        q_Z : ArrayLike
        K_R : ArrayLike
        K_zeta : ArrayLike
        K_Z : ArrayLike

        Returns
        -------
        ArrayLike

        """

        K_magnitude = np.sqrt(K_X**2 + K_Y**2 + K_Z**2)
        poloidal_flux = self.field.poloidal_flux(q_X, q_Y, q_Z)
        electron_density = self.density(poloidal_flux)

        if self.temperature:
            temperature = self.temperature(poloidal_flux)
        else:
            temperature = None

        B_X = np.squeeze(self.field.B_X(q_X, q_Y, q_Z))
        B_Y = np.squeeze(self.field.B_Y(q_X, q_Y, q_Z))
        B_Z = np.squeeze(self.field.B_Z(q_X, q_Y, q_Z))

        B_total = np.sqrt(B_X**2 + B_Y**2 + B_Z**2)
        b_hat = np.array([B_X, B_Y, B_Z]) / B_total
        K_hat = np.array([K_X, K_Y, K_Z]) / K_magnitude

        # square of the mismatch angle
        if np.size(q_X) == 1:
            sin_theta_m_sq = np.dot(b_hat, K_hat) ** 2
        else:  # Vectorised version of find_H
            b_hat = b_hat.T
            K_hat = K_hat.T
            sin_theta_m_sq = dot(b_hat, K_hat) ** 2

        epsilon = DielectricTensor(
            electron_density, self.angular_frequency, B_total, temperature
        )

        Booker_alpha = (epsilon.e_bb * sin_theta_m_sq) + epsilon.e_11 * (
            1 - sin_theta_m_sq
        )
        Booker_beta = (-epsilon.e_11 * epsilon.e_bb * (1 + sin_theta_m_sq)) - (
            epsilon.e_11**2 - epsilon.e_12**2
        ) * (1 - sin_theta_m_sq)
        Booker_gamma = epsilon.e_bb * (epsilon.e_11**2 - epsilon.e_12**2)

        H_discriminant = np.maximum(
            np.zeros_like(Booker_beta),
            (Booker_beta**2 - 4 * Booker_alpha * Booker_gamma),
        )

        return (K_magnitude / self.wavenumber_K0) ** 2 + (
            Booker_beta - self.mode_flag * np.sqrt(H_discriminant)
        ) / (2 * Booker_alpha)

    def derivatives(
        self,
        q_X: ArrayLike,
        q_Y: ArrayLike,
        q_Z: ArrayLike,
        K_X: ArrayLike,
        K_Y: ArrayLike,
        K_Z: ArrayLike,
        second_order: bool = False,
    ) -> Dict[str, ArrayLike]:
        """Evaluate the first-order derivative in all directions at the given
        point(s), and optionally the second-order ones too

        Parameters
        ----------
        q_R : ArrayLike
        q_Z : ArrayLike
        K_R : ArrayLike
        K_zeta : ArrayLike
        K_Z : ArrayLike
            Coordinates to evaluate the derivatives at
        second_order : bool
            If ``True``, also evaluate the second derivatives

        """

        # Capture the location we want the derivatives at
        starts = {
            "q_X": q_X,
            "q_Y": q_Y,
            "q_Z": q_Z,
            "K_X": K_X,
            "K_Y": K_Y,
            "K_Z": K_Z,
        }

        def apply_stencil(dims: Tuple[str, ...], stencil: str):
            return derivative(self, dims, starts, self.spacings, stencil)

        # We always compute the first order derivatives
        derivatives = {
            "dH_dX": apply_stencil(("q_X",), "d1_FFD2"),
            "dH_dY": apply_stencil(("q_Y",), "d1_FFD2"),
            "dH_dZ": apply_stencil(("q_Z",), "d1_FFD2"),
            "dH_dKX": apply_stencil(("K_X",), "d1_CFD2"),
            "dH_dKY": apply_stencil(("K_Y",), "d1_CFD2"),
            "dH_dKZ": apply_stencil(("K_Z",), "d1_CFD2"),
        }

        if second_order:
            second_derivatives = {
                "d2H_dX2": apply_stencil(("q_X", "q_X"), "d2_FFD2"),
                "d2H_dY2": apply_stencil(("q_Y", "q_Y"), "d2_FFD2"),
                "d2H_dZ2": apply_stencil(("q_Z", "q_Z"), "d2_FFD2"),
                "d2H_dKX2": apply_stencil(("K_X", "K_X"), "d2_CFD2"),
                "d2H_dKY2": apply_stencil(("K_Y", "K_Y"), "d2_CFD2"),
                "d2H_dKZ2": apply_stencil(("K_Z", "K_Z"), "d2_CFD2"),
                "d2H_dX_dY": apply_stencil(("q_X", "q_Y"), "d1d1_FFD_FFD2"),
                "d2H_dX_dZ": apply_stencil(("q_X", "q_Z"), "d1d1_FFD_FFD2"),
                "d2H_dY_dZ": apply_stencil(("q_Y", "q_Z"), "d1d1_FFD_FFD2"),
                # "d2H_dY_dX": apply_stencil(("q_Y", "q_X"), "d1d1_FFD_FFD2"),
                # "d2H_dZ_dX": apply_stencil(("q_Z", "q_X"), "d1d1_FFD_FFD2"),
                # "d2H_dZ_dY": apply_stencil(("q_Z", "q_Y"), "d1d1_FFD_FFD2"),
                "d2H_dX_dKX": apply_stencil(("q_X", "K_X"), "d1d1_FFD_CFD2"),
                "d2H_dX_dKY": apply_stencil(("q_X", "K_Y"), "d1d1_FFD_CFD2"),
                "d2H_dX_dKZ": apply_stencil(("q_X", "K_Z"), "d1d1_FFD_CFD2"),
                "d2H_dY_dKX": apply_stencil(("q_Y", "K_X"), "d1d1_FFD_CFD2"),
                "d2H_dY_dKY": apply_stencil(("q_Y", "K_Y"), "d1d1_FFD_CFD2"),
                "d2H_dY_dKZ": apply_stencil(("q_Y", "K_Z"), "d1d1_FFD_CFD2"),
                "d2H_dZ_dKX": apply_stencil(("q_Z", "K_X"), "d1d1_FFD_CFD2"),
                "d2H_dZ_dKY": apply_stencil(("q_Z", "K_Y"), "d1d1_FFD_CFD2"),
                "d2H_dZ_dKZ": apply_stencil(("q_Z", "K_Z"), "d1d1_FFD_CFD2"),
                "d2H_dKX_dKZ": apply_stencil(("K_X", "K_Z"), "d1d1_CFD_CFD2"),
                "d2H_dKX_dKY": apply_stencil(("K_X", "K_Y"), "d1d1_CFD_CFD2"),
                "d2H_dKY_dKZ": apply_stencil(("K_Y", "K_Z"), "d1d1_CFD_CFD2"),
                # "d2H_dKZ_dKX": apply_stencil(("K_Z", "K_X"), "d1d1_CFD_CFD2"),
                # "d2H_dKY_dKX": apply_stencil(("K_Y", "K_X"), "d1d1_CFD_CFD2"),
                # "d2H_dKZ_dKY": apply_stencil(("K_Z", "K_Y"), "d1d1_CFD_CFD2"),
            }
            derivatives.update(second_derivatives)

        return derivatives


def hessians(dH: dict):
    r"""Compute the elements of the Hessian of the Hamiltonian:

    .. math::
          \begin{gather}
            \nabla \nabla H \\
            \nabla_K \nabla H \\
            \nabla_K \nabla_K H \\
          \end{gather}

    given a ``dict`` containing the second derivatives of the Hamiltonian (as
    returned from `Hamiltonian.derivatives` with ``second_order=True``)
    """
    d2H_dX2 = dH["d2H_dX2"]
    d2H_dY2 = dH["d2H_dY2"]
    d2H_dZ2 = dH["d2H_dZ2"]
    d2H_dKX2 = dH["d2H_dKX2"]
    d2H_dKY2 = dH["d2H_dKY2"]
    d2H_dKZ2 = dH["d2H_dKZ2"]
    d2H_dX_dY = dH["d2H_dX_dY"]
    d2H_dX_dZ = dH["d2H_dX_dZ"]
    d2H_dY_dZ = dH["d2H_dY_dZ"]
    # d2H_dY_dX = dH["d2H_dY_dX"]
    # d2H_dZ_dX = dH["d2H_dZ_dX"]
    # d2H_dZ_dY = dH["d2H_dZ_dY"]
    d2H_dKX_dX = dH["d2H_dX_dKX"]
    d2H_dKY_dX = dH["d2H_dX_dKY"]
    d2H_dKZ_dX = dH["d2H_dX_dKZ"]
    d2H_dKX_dY = dH["d2H_dY_dKX"]
    d2H_dKY_dY = dH["d2H_dY_dKY"]
    d2H_dKZ_dY = dH["d2H_dY_dKZ"]
    d2H_dKX_dZ = dH["d2H_dZ_dKX"]
    d2H_dKY_dZ = dH["d2H_dZ_dKY"]
    d2H_dKZ_dZ = dH["d2H_dZ_dKZ"]
    d2H_dKX_dKY = dH["d2H_dKX_dKY"]
    d2H_dKX_dKZ = dH["d2H_dKX_dKZ"]
    d2H_dKY_dKZ = dH["d2H_dKY_dKZ"]
    # d2H_dKY_dKX = dH["d2H_dKY_dKX"]
    # d2H_dKZ_dKX = dH["d2H_dKZ_dKX"]
    # d2H_dKZ_dKY = dH["d2H_dKZ_dKY"]

    zeros = np.zeros_like(d2H_dX2)

    def reshape(array: FloatArray):
        """Such that shape is [points,3,3] instead of [3,3,points]"""
        if array.ndim == 2:
            return array
        return np.moveaxis(np.squeeze(array), 2, 0)

    grad_grad_H = reshape(
        np.array(
            [
                [d2H_dX2, d2H_dX_dY, d2H_dX_dZ],
                [d2H_dX_dY, d2H_dY2, d2H_dY_dZ],
                [d2H_dX_dZ, d2H_dY_dZ, d2H_dZ2],
            ]
        )
    )
    gradK_grad_H = reshape(
        np.array(
            [
                [d2H_dKX_dX, d2H_dKX_dY, d2H_dKX_dZ],
                [d2H_dKY_dX, d2H_dKY_dY, d2H_dKY_dZ],
                [d2H_dKZ_dX, d2H_dKZ_dY, d2H_dKZ_dZ],
            ]
        )
    )
    gradK_gradK_H = reshape(
        np.array(
            [
                [d2H_dKX2, d2H_dKX_dKY, d2H_dKX_dKZ],
                [d2H_dKX_dKY, d2H_dKY2, d2H_dKY_dKZ],
                [d2H_dKX_dKZ, d2H_dKY_dKZ, d2H_dKZ2],
            ]
        )
    )
    return grad_grad_H, gradK_grad_H, gradK_gradK_H
