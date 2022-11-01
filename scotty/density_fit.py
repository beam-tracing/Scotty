from __future__ import annotations
from typing import Optional
from warnings import warn

from scotty.typing import PathLike

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import UnivariateSpline


class DensityFit:
    """Base class for density parameterisations.

    Subclasses should implement ``_fit_impl`` which takes a 1D array
    of the poloidal flux and returns the density at those points. This
    base class will handle setting the density to zero outside the
    plasma.

    Parameters
    ==========
    poloidal_flux_enter:
        Flux at edge of plasma. Density is zero outside this flux label

    """

    def __init__(self, poloidal_flux_enter: float):
        self.poloidal_flux_enter = poloidal_flux_enter

    def __call__(self, poloidal_flux: ArrayLike) -> ArrayLike:
        """Returns the interpolated density at ``poloidal_flux`` points."""
        density = self._fit_impl(poloidal_flux)
        # Mask density inside plasma
        is_inside = poloidal_flux <= self.poloidal_flux_enter
        return is_inside * density

    def _fit_impl(self, poloidal_flux: ArrayLike) -> ArrayLike:
        raise NotImplementedError


class QuadraticFit(DensityFit):
    r"""Quadratic fit

    Given the density on the magnetic axis, :math:`n_{e0}`, and the
    poloidal flux where the density goes to zero, :math:`\psi_0`, the
    density is given by:

    .. math::

        n_e(\psi) = n_{e0} - \frac{n_{e0}}{\psi_0}\psi^2

    Parameters
    ==========
    poloidal_flux_enter:
        Poloidal flux where density goes to zero (:math:`\psi_0` above)
    ne_0:
        Density at magnetic axis (:math:`n_{e0} \equiv n_e(\psi = 0)`)
    psi_0:
        If passed, this must be the same as ``poloidal_flux_enter``

        .. deprecated:: 2.4.0
           ``psi_0`` can be safely dropped
    """

    def __init__(
        self, poloidal_flux_enter: float, ne_0: float, psi_0: Optional[float] = None
    ):
        super().__init__(poloidal_flux_enter)
        self.ne_0 = ne_0

        if psi_0 is not None:
            warn(
                "'psi_0' argument to `QuadraticFit` is deprecated and can be removed",
                DeprecationWarning,
            )
            if psi_0 != poloidal_flux_enter:
                raise ValueError(
                    f"QuadraticFit: 'psi_0' ({psi_0}) doesn't agree with "
                    f"'poloidal_flux_enter' ({poloidal_flux_enter})"
                )

    def _fit_impl(self, poloidal_flux: ArrayLike) -> ArrayLike:
        return self.ne_0 - ((self.ne_0 / self.poloidal_flux_enter) * poloidal_flux**2)


class TanhFit(DensityFit):
    r"""Fit using :math:`\tanh`:

    .. math::

        n_e(\psi) = n_{e0} \tanh\left(n_{e1} (\psi - \psi_0)\right)

    Parameters
    ==========
    poloidal_flux_enter:
        Poloidal flux where density goes to zero (:math:`\psi_0` above)
    ne_0:
        (Asymptotic) density at magnetic axis (:math:`n_{e0} \le n_e(\psi = 0)`)
    ne_1:
        Second fitting parameter
    psi_0:
        If passed, this must be the same as ``poloidal_flux_enter``

        .. deprecated:: 2.4.0
           ``psi_0`` can be safely dropped

    """

    def __init__(
        self,
        poloidal_flux_enter: float,
        ne_0: float,
        ne_1: float,
        psi_0: Optional[float] = None,
    ):
        super().__init__(poloidal_flux_enter)
        self.ne_0 = ne_0
        self.ne_1 = ne_1

        if psi_0 is not None:
            warn(
                "'psi_0' argument to `QuadraticFit` is deprecated and can be removed",
                DeprecationWarning,
            )
            if psi_0 != poloidal_flux_enter:
                raise ValueError(
                    f"QuadraticFit: 'psi_0' ({psi_0}) doesn't agree with "
                    f"'poloidal_flux_enter' ({poloidal_flux_enter})"
                )

    def _fit_impl(self, poloidal_flux: ArrayLike) -> ArrayLike:
        return self.ne_0 * np.tanh(
            self.ne_1 * (poloidal_flux - self.poloidal_flux_enter)
        )


class PolynomialFit(DensityFit):
    r"""Fit using a :math:`m`-th order polynomial:

    .. math::

        n_e(\psi) = \sum_{i=0,m} a_i \psi^i

    where :math:`m+1` is the number of coefficients

    Parameters
    ==========
    poloidal_flux_enter:
        Poloidal flux where density goes to zero (:math:`\psi_0` above)
    coefficients:
        List of polynomial coefficients, from highest degree to the
        constant term
    """

    def __init__(self, poloidal_flux_enter: float, *coefficients):
        super().__init__(poloidal_flux_enter)
        self.coefficients = coefficients

    def _fit_impl(self, poloidal_flux: ArrayLike) -> ArrayLike:
        return np.polyval(self.coefficients, poloidal_flux)


class StefanikovaFit(DensityFit):
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
        poloidal_flux_enter: float,
        a_height: float,
        a_width: float,
        a_exp: float,
        b_height: float,
        b_SOL: float,
        b_width: float,
        b_slope: float,
        b_pos: float,
    ):
        super().__init__(poloidal_flux_enter)
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
        fp = self._f_ped(poloidal_flux)
        return (
            fp
            + (self.a_height - fp) * np.exp(-poloidal_flux / self.a_width) ** self.a_exp
        )


class SmoothingSplineFit(DensityFit):
    """1D smoothing spline using `scipy.interpolate.UnivariateSpline`"""

    def __init__(
        self,
        poloidal_flux_enter: float,
        poloidal_flux: ArrayLike,
        density: ArrayLike,
        order: int = 5,
        smoothing: Optional[float] = None,
    ):
        super().__init__(poloidal_flux_enter)
        self.spline = UnivariateSpline(
            poloidal_flux,
            density,
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
        filename: PathLike,
        poloidal_flux_enter: float,
        order: int = 5,
        smoothing: Optional[float] = None,
    ) -> SmoothingSplineFit:
        r"""Create a `SmoothingSplineFit` using parameters from text
        file.

        File should contain electron density as a function of poloidal
        flux label in units of :math:`10^{19} \mathrm{m}^{-3}`.

        The first line is ignored, the rest of the file is expected to
        be in two columns: the first should contain the radial
        coordinate (:math:`\sqrt(\psi)`), the second the electron
        density. The columns should be separated by whitespace.

        .. code-block:: text

            101
            0.00000000e+00 4.00000000e+00
            1.00000000e-02 3.99960000e+00
            2.00000000e-02 3.99840000e+00
            ...

        """

        ne_data = np.fromfile(filename, dtype=float, sep="   ")
        radialcoord_array = ne_data[1::2]
        density_array = ne_data[2::2]
        # Loading radial coord for now, makes it easier to benchmark
        # with Torbeam. Hence, have to convert to poloidal flux
        poloidal_flux_array = radialcoord_array**2

        return cls(
            poloidal_flux_enter, poloidal_flux_array, density_array, order, smoothing
        )

    def _fit_impl(self, poloidal_flux):
        return self.spline(poloidal_flux)


##################################################


DENSITY_FIT_METHODS = {
    "smoothing-spline": SmoothingSplineFit,
    "smoothing-spline-file": SmoothingSplineFit.from_dat_file,
    "stefanikova": StefanikovaFit,
    "poly3": PolynomialFit,
    "polynomial": PolynomialFit,
    "tanh": TanhFit,
    "quadratic": QuadraticFit,
}


def density_fit(
    method: Optional[str],
    poloidal_flux_enter: float,
    parameters: ArrayLike,
    filename: Optional[PathLike] = None,
) -> DensityFit:
    """Create a density profile parameterisation

    Parameters
    ==========
    method:
        Name of density fit parameterisation
    poloidal_flux_enter:
        Poloidal flux label where density goes to zero
    parameters:
        List of parameters passed to density fit
    filename:


    """

    if method is None:
        warn(
            "Creating a density fit without specifying a method is deprecated. "
            "Guessing method from parameters",
            DeprecationWarning,
        )
        if filename is not None:
            print("ne(psi): loading from input file")
            return SmoothingSplineFit.from_dat_file(
                filename, poloidal_flux_enter, *parameters
            )

        FIT_LENGTH_MAPPINGS = {
            8: ("Stefanikova", StefanikovaFit),
            4: ("order_3_polynomial", PolynomialFit),
            3: ("constant*tanh", TanhFit),
            2: ("quadratic profile", QuadraticFit),
        }

        try:
            name, fit_method = FIT_LENGTH_MAPPINGS[len(parameters)]
            print(f"ne(psi): Using {name}")
        except KeyError:
            raise ValueError(
                "Couldn't match length of 'parameters' to known fit parameterisation. "
                "Try explicitly providing fit method"
            )
    else:
        try:
            fit_method = DENSITY_FIT_METHODS[method]
        except KeyError:
            raise ValueError(
                f"Unknown density fitting method '{method}'. "
                f"Expected one of {DENSITY_FIT_METHODS.keys()}"
            )

    return fit_method(poloidal_flux_enter, *parameters)
