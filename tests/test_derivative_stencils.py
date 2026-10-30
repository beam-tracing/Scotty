"""Order of accuracy of every finite-difference stencil in scotty.derivatives.

For a smooth function, halving the step must divide the error by about 2**order,
where the order is the trailing digit of the stencil name.
"""

from scotty.derivatives import STENCILS, derivative

import numpy as np
import pytest


def f(x, y):
    return np.sin(x) * np.cos(2 * y) + 0.3 * x**3 * y


X0, Y0 = 0.4, -0.7
EXACT = {
    "d1": np.cos(X0) * np.cos(2 * Y0) + 0.9 * X0**2 * Y0,  # df/dx
    "d2": -np.sin(X0) * np.cos(2 * Y0) + 1.8 * X0 * Y0,  # d2f/dx2
    "d1d1": -2 * np.cos(X0) * np.sin(2 * Y0) + 0.9 * X0**2,  # d2f/dxdy
}
DIMS = {"d1": "x", "d2": ("x", "x"), "d1d1": ("x", "y")}


@pytest.mark.parametrize("stencil", list(STENCILS))
def test_stencil_order_of_accuracy(stencil):
    kind = stencil.split("_")[0]
    order = int(stencil[-1])

    def error(h):
        value = derivative(
            f,
            DIMS[kind],
            {"x": X0, "y": Y0},
            spacings=h,
            stencil=stencil,
            use_cache=False,
        )
        return abs(value - EXACT[kind])

    observed_order = np.log2(error(1e-2) / error(5e-3))
    assert (
        observed_order > order - 0.3
    ), f"{stencil}: observed order {observed_order:.2f}, expected {order}"


@pytest.mark.xfail(
    strict=True,
    reason="derivative() does not check the stencil's dimensionality against dims, "
    "so a 1D stencil with two dims silently truncates (derivatives.py)",
)
def test_stencil_dimension_mismatch_raises():
    with pytest.raises(ValueError):
        derivative(
            f,
            ("x", "y"),
            {"x": X0, "y": Y0},
            spacings=1e-3,
            stencil="d1_CFD2",
            use_cache=False,
        )
