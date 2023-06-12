# Copyright 2023 - 2023, Valerian Hall-Chen and the Scotty contributors
# SPDX-License-Identifier: GPL-3.0

from __future__ import annotations
from typing import Callable, Optional, List, Dict, Union, Sequence
from warnings import warn

from scotty.typing import PathLike, ArrayLike

import numpy as np
from scipy.interpolate import UnivariateSpline


ProfileFitLike = Callable[[ArrayLike], ArrayLike]
"""A callable that can parameterise density/temperature in 1D"""


class ProfileFit:
    """Base class for parameterisation of density/temperature profiles.

    Subclasses should implement ``_fit_impl`` which takes a 1D array
    of the poloidal flux and returns the density or temperature at those points. This
    base class will handle setting the profile to zero outside the
    plasma.

    Parameters
    ==========
    poloidal_flux_zero_profile:
        Flux at edge of plasma. Profile is zero outside this flux label.

    """

    def __init__(self, poloidal_flux_zero_profile: float):
        self.poloidal_flux_zero_profile = poloidal_flux_zero_profile

    def __call__(self, poloidal_flux: ArrayLike) -> ArrayLike:
        """Returns the interpolated profile at ``poloidal_flux`` points."""

        poloidal_flux = np.asfarray(poloidal_flux)
        profile = np.asfarray(self._fit_impl(poloidal_flux))
        # Mask density inside plasma
        is_inside = poloidal_flux <= self.poloidal_flux_zero_profile
        return is_inside * profile

    def _fit_impl(self, poloidal_flux: ArrayLike) -> ArrayLike:
        raise NotImplementedError


class LinearFit(ProfileFit):
    r""" "Linear fit

    Currently implemented as a test profile for temperature.

    For the sake of convenience, the variable ne_0 is named for electron
    density but the class works for electron temperature as well.
    """

    def __init__(
        self,
        poloidal_flux_zero_profile: float,
        ne_0: float,  # Core temperature/density
        psi_0: Optional[float] = None,
    ):
        super().__init__(poloidal_flux_zero_profile)
        self.ne_0 = ne_0

        if psi_0 is not None:
            warn(
                "'psi_0' argument to `LinearFit` is deprecated and can be removed",
                DeprecationWarning,
            )
            if psi_0 != poloidal_flux_zero_profile:
                raise ValueError(
                    f"LinearFit: 'psi_0' ({psi_0}) doesn't agree with "
                    f"'poloidal_flux_zero_(density or temperature)' ({poloidal_flux_zero_profile})"
                )

    def _fit_impl(self, poloidal_flux: ArrayLike) -> ArrayLike:
        poloidal_flux = np.asfarray(poloidal_flux)
        return self.ne_0 - (
            (self.ne_0 / self.poloidal_flux_zero_profile) * poloidal_flux
        )

    def __repr__(self):
        return f"LinearFit({self.poloidal_flux_zero_profile}, core_val={self.ne_0})"


class QuadraticFit(ProfileFit):
    r"""Quadratic fit

    Given the densit on the magnetic axis, :math:`n_{e0}`, and the
    poloidal flux where the density goes to zero, :math:`\psi_0`, the
    density is given by:

    .. math::

        n_e(\psi) = n_{e0} - \frac{n_{e0}}{\psi_0}\psi^2

    For the sake of convenience, the variable ne_0 is named for electron
    density but the class works for electron temperature as well.

    Parameters
    ==========
    poloidal_flux_zero_profile:
        Poloidal flux where profile goes to zero (:math:`\psi_0` above)
    ne_0:
        Profile value at magnetic axis (e.g. for density, :math:`n_{e0} \equiv n_e(\psi = 0)`)
    psi_0:
        If passed, this must be the same as ``poloidal_flux_zero_profile``

        .. deprecated:: 2.4.0
           ``psi_0`` can be safely dropped
    """

    def __init__(
        self,
        poloidal_flux_zero_profile: float,
        ne_0: float,
        psi_0: Optional[float] = None,
    ):
        super().__init__(poloidal_flux_zero_profile)
        self.ne_0 = ne_0

        if psi_0 is not None:
            warn(
                "'psi_0' argument to `QuadraticFit` is deprecated and can be removed",
                DeprecationWarning,
            )
            if psi_0 != poloidal_flux_zero_profile:
                raise ValueError(
                    f"QuadraticFit: 'psi_0' ({psi_0}) doesn't agree with "
                    f"'poloidal_flux_zero_profile' ({poloidal_flux_zero_profile})"
                )

    def _fit_impl(self, poloidal_flux: ArrayLike) -> ArrayLike:
        poloidal_flux = np.asfarray(poloidal_flux)
        return self.ne_0 - (
            (self.ne_0 / self.poloidal_flux_zero_profile) * poloidal_flux**2
        )

    def __repr__(self):
        return f"QuadraticFit({self.poloidal_flux_zero_profile}, core_val={self.ne_0})"


class TanhFit(ProfileFit):
    r"""Fit using :math:`\tanh`:

    .. math::

        n_e(\psi) = n_{e0} \tanh\left(n_{e1} (\psi - \psi_0)\right)

    Parameters
    ==========
    poloidal_flux_zero_profile:
        Poloidal flux where profile goes to zero (:math:`\psi_0` above)
    ne_0:
        (Asymptotic) density at magnetic axis (:math:`n_{e0} \le n_e(\psi = 0)`)
    ne_1:
        Second fitting parameter
    psi_0:
        If passed, this must be the same as ``poloidal_flux_zero_profile``

        .. deprecated:: 2.4.0
           ``psi_0`` can be safely dropped

    """

    def __init__(
        self,
        poloidal_flux_zero_profile: float,
        ne_0: float,
        ne_1: float,
        psi_0: Optional[float] = None,
    ):
        super().__init__(poloidal_flux_zero_profile)
        self.ne_0 = ne_0
        self.ne_1 = ne_1

        if psi_0 is not None:
            warn(
                "'psi_0' argument to `QuadraticFit` is deprecated and can be removed",
                DeprecationWarning,
            )
            if psi_0 != poloidal_flux_zero_profile:
                raise ValueError(
                    f"QuadraticFit: 'psi_0' ({psi_0}) doesn't agree with "
                    f"'poloidal_flux_zero_profile' ({poloidal_flux_zero_profile})"
                )

    def _fit_impl(self, poloidal_flux: ArrayLike) -> ArrayLike:
        poloidal_flux = np.asfarray(poloidal_flux)
        return self.ne_0 * np.tanh(
            self.ne_1 * (poloidal_flux - self.poloidal_flux_zero_profile)
        )

    def __repr__(self):
        return f"TanhFit({self.poloidal_flux_zero_profile}, ne_0={self.ne_0}, ne_1={self.ne_1})"


class PolynomialFit(ProfileFit):
    r"""Fit using a :math:`m`-th order polynomial:

    .. math::

        n_e(\psi) = \sum_{i=0,m} a_i \psi^i

    where :math:`m+1` is the number of coefficients

    Parameters
    ==========
    poloidal_flux_zero_profile:
        Poloidal flux where solver starts and/or boundary conditions are applied; density has to be zero in the current implementation (:math:`\psi_0` above)
    coefficients:
        List of polynomial coefficients, from highest degree to the constant term
    """

    def __init__(self, poloidal_flux_zero_profile: float, *coefficients):
        super().__init__(poloidal_flux_zero_profile)
        self.coefficients = coefficients

    def _fit_impl(self, poloidal_flux: ArrayLike) -> ArrayLike:
        poloidal_flux = np.asfarray(poloidal_flux)
        return np.polyval(self.coefficients, poloidal_flux)

    def __repr__(self):
        return f"PolynomialFit({self.poloidal_flux_zero_profile}, {', '.join(self.coefficients)})"


class StefanikovaFit(ProfileFit):
    r"""Fit according to Stefanikova et al (2016) [1]_

    .. math::

        F_\mathrm{ped}(r, b) &= \frac{b_\mathrm{height} − b_\mathrm{SOL}}{2}
            \left[
                \mathrm{mtanh}\left(
                    \frac{b_\mathrm{pos} − r}{2 b_\mathrm{width}}, b_\mathrm{slope}
                \right) + 1
            \right] + b_\mathrm{SOL} \\
        \mathrm{mtanh}(x, b_\mathrm{slope}) &=
            \frac{(1+b_\mathrm{slope}x)e^x - e^{−x}}{e^x+e^{−x}} \\
        F_\mathrm{full}(r, a, b) &= F_\mathrm{ped}(r, b)
            + \left[ a_\mathrm{height} - F_\mathrm{ped}(r, b) \right]
            \cdot \exp\left(-(\frac{r}{a_\mathrm{width}})^{a_\mathrm{exp}}\right)

    Adapted from code by Simon Freethy

    .. rubric:: Footnotes

    .. [1] https://doi.org/10.1063/1.4961554
    """

    def __init__(
        self,
        poloidal_flux_zero_profile: float,
        a_height: float,
        a_width: float,
        a_exp: float,
        b_height: float,
        b_SOL: float,
        b_width: float,
        b_slope: float,
        b_pos: float,
    ):
        super().__init__(poloidal_flux_zero_profile)
        self.a_height = a_height
        self.a_width = a_width
        self.a_exp = a_exp
        self.b_height = b_height
        self.b_SOL = b_SOL
        self.b_width = b_width
        self.b_slope = b_slope
        self.b_pos = b_pos

    def _mtanh(self, x):
        e_x = np.exp(x)
        e_mx = np.exp(-x)
        return ((1 + self.b_slope * x) * e_x - e_mx) / (e_x + e_mx)

    def _f_ped(self, r):
        mth = self._mtanh((self.b_pos - r) / (2 * self.b_width))
        return (self.b_height - self.b_SOL) / 2 * (mth + 1) + self.b_SOL

    def _fit_impl(self, poloidal_flux: ArrayLike) -> ArrayLike:
        poloidal_flux = np.asfarray(poloidal_flux)
        fp = self._f_ped(poloidal_flux)
        return (
            fp
            + (self.a_height - fp) * np.exp(-poloidal_flux / self.a_width) ** self.a_exp
        )

    def __repr__(self):
        return (
            f"StefanikovaFit({self.poloidal_flux_zero_profile}, "
            f"a_height={self.a_height}, "
            f"a_width={self.a_width}, "
            f"a_exp={self.a_exp}, "
            f"b_height={self.b_height}, "
            f"b_SOL={self.b_SOL}, "
            f"b_width={self.b_width}, "
            f"b_slope={self.b_slope}, "
            f"b_pos={self.b_pos})"
        )


class SmoothingSplineFit(ProfileFit):
    """1D smoothing spline using `scipy.interpolate.UnivariateSpline`"""

    def __init__(
        self,
        poloidal_flux_zero_profile: float,
        poloidal_flux: ArrayLike,
        fitting_data: ArrayLike,
        order: int = 5,
        smoothing: Optional[float] = None,
    ):
        super().__init__(poloidal_flux_zero_profile)
        self.spline = UnivariateSpline(
            poloidal_flux,
            fitting_data,
            w=None,
            bbox=[None, None],
            k=order,
            s=smoothing,
            ext=0,
            check_finite=False,
        )

    @classmethod
    def from_dat_file(
        cls,
        poloidal_flux_zero_profile: float,
        filename: PathLike,
        order: int = 5,
        smoothing: Optional[float] = None,
    ) -> SmoothingSplineFit:
        r"""Create a `SmoothingSplineFit` using parameters from text
        file.

        File should contain electron density/temperature as a function of poloidal
        flux label in units of :math:`10^{19} \mathrm{m}^{-3}`.

        The first line is ignored, the rest of the file is expected to
        be in two columns: the first should contain the radial
        coordinate (:math:`\sqrt(\psi)`), the second the electron
        density/temperature. The columns should be separated by whitespace.

        .. code-block:: text

            101
            0.00000000e+00 4.00000000e+00
            1.00000000e-02 3.99960000e+00
            2.00000000e-02 3.99840000e+00
            ...

        """

        fitting_data = np.fromfile(filename, dtype=float, sep="   ")
        radialcoord_array = fitting_data[1::2]
        fitting_array = fitting_data[2::2]
        # Loading radial coord for now, makes it easier to benchmark
        # with Torbeam. Hence, have to convert to poloidal flux
        poloidal_flux_array = radialcoord_array**2

        return cls(
            poloidal_flux_zero_profile,
            poloidal_flux_array,
            fitting_array,
            order,
            smoothing,
        )

    def _fit_impl(self, poloidal_flux):
        return self.spline(poloidal_flux)


##################################################


PROFILE_FIT_METHODS: Dict[str, Union[type, Callable]] = {
    "smoothing-spline": SmoothingSplineFit,
    "smoothing-spline-file": SmoothingSplineFit.from_dat_file,
    "stefanikova": StefanikovaFit,
    "poly3": PolynomialFit,
    "polynomial": PolynomialFit,
    "tanh": TanhFit,
    "quadratic": QuadraticFit,
    "linear": LinearFit,
}


def _guess_profile_fit_method(
    parameters: Sequence, filename: Optional[PathLike]
) -> str:
    if filename is not None:
        print("ne(psi): loading from input file")
        return "smoothing-spline-file"

    FIT_LENGTH_MAPPINGS = {
        8: ("Stefanikova", "stefanikova"),
        4: ("order_3_polynomial", "polynomial"),
        3: ("constant*tanh", "tanh"),
        2: ("quadratic profile", "quadratic"),
    }

    try:
        name, fit_method = FIT_LENGTH_MAPPINGS[len(parameters)]
        print(f"ne(psi): Using {name}")
        return fit_method
    except KeyError:
        raise ValueError(
            "Couldn't match length of 'parameters' to known fit parameterisation. "
            "Try explicitly providing fit method"
        )


def profile_fit(
    method: Optional[str],
    poloidal_flux_zero_profile: float,
    parameters: Sequence,
    filename: Optional[PathLike] = None,
) -> ProfileFit:
    """Create a density/temperature profile parameterisation

    Parameters
    ==========
    method:
        Name of profile fit parameterisation
    poloidal_flux_zero_profile:
        Poloidal flux label where density/temperature goes to zero
    parameters:
        List of parameters passed to profile fit
    filename:


    """

    if method is None:
        method = _guess_profile_fit_method(parameters, filename)
        warn(
            "Creating a density fit without specifying a method is deprecated. "
            f"Guessing method '{method}' from parameters.\n"
            f"\tPass `density_fit_method='{method}'` to `beam_me_up` to suppress this warning",
            DeprecationWarning,
        )

    if method == "smoothing-spline-file" and filename != parameters[0]:
        raise ValueError(
            f"For '{method}', expected filename as second parameter, but got '{parameters[0]}'"
        )

    try:
        fit_method = PROFILE_FIT_METHODS[method]
    except KeyError:
        raise ValueError(
            f"Unknown density fitting method '{method}'. "
            f"Expected one of {PROFILE_FIT_METHODS.keys()}"
        )

    return fit_method(poloidal_flux_zero_profile, *parameters)
