from scotty.density_fit import (
    QuadraticFit,
    TanhFit,
    PolynomialFit,
    StefanikovaFit,
    SmoothingSplineFit,
)

import numpy as np

import pytest

# Parameters for density fits
CENTRAL_DENSITY = 2.0
LCFS = 1.0


@pytest.mark.parametrize(
    "fit",
    [
        pytest.param(QuadraticFit(LCFS, CENTRAL_DENSITY), id="QuadraticFit"),
        pytest.param(TanhFit(LCFS, CENTRAL_DENSITY, -CENTRAL_DENSITY), id="TanhFit"),
        pytest.param(
            PolynomialFit(LCFS, -3.1, 3.3, -1.55, CENTRAL_DENSITY), id="PolynomialFit"
        ),
        pytest.param(
            StefanikovaFit(LCFS, CENTRAL_DENSITY, 1.0, 1.2, 1.0, 1.1, 0.8, 0.3, 0.9),
            id="StefanikovaFit",
        ),
        pytest.param(
            SmoothingSplineFit(
                LCFS, np.linspace(0, 1, 10), np.linspace(CENTRAL_DENSITY, 0.0, 10)
            ),
            id="SmoothingSplineFit",
        ),
    ],
)
def test_density_fit(fit):
    # Not all fitting functions capture the density on the axis very well
    assert np.isclose(fit(0), CENTRAL_DENSITY, atol=0.1), "Axis"
    assert fit(LCFS + 0.1) == 0.0, "Outside"

    density = fit(np.linspace(0, LCFS, 10))
    assert np.all(density[:-1] > density[1:]), "Monotonically decreasing"


def test_spline_fit_from_file(tmp_path):
    rho = np.linspace(0, 1, 11)
    density = rho**2

    filename = tmp_path / "ne.dat"
    with open(filename, "w") as f:
        f.write(f"{len(rho)}\n")
        np.savetxt(f, np.column_stack((rho, density)), fmt="%.7e")

    fit = SmoothingSplineFit.from_dat_file(filename, 1.0)

    assert np.allclose(density, fit(rho**2))
