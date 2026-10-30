"""Tests and benchmarks for the analytical magnetic fields with circular flux
surfaces, comparing their exact poloidal flux derivatives with finite
differences, plus checks that the field base classes are abstract.

Run with ``pytest -s tests/test_analytical_fields.py`` to see the timing
benchmark printed."""

from scotty.geometry import (
    CircularCrossSectionField,
    CircularFieldBase,
    CircularFluxSurfaceField,
    ConstantCurrentDensityField,
    MagneticField_Cartesian,
    MagneticField_Cylindrical,
    create_magnetic_geometry,
)

import numpy as np
from numpy.testing import assert_allclose
import pathlib
import pytest
import time

B_T_AXIS = 1.0
R_AXIS = 1.5
MINOR_RADIUS_A = 0.5
B_P_A = 0.1

# Step size used by default in beam_me_up
DELTA = 1e-4

FIELD_CLASSES = [
    pytest.param(CircularCrossSectionField, id="circular-cross-section"),
    pytest.param(ConstantCurrentDensityField, id="constant-current-density"),
    pytest.param(CircularFluxSurfaceField, id="circular-flux-surface"),
]

# (method name, number of step sizes it takes, tolerance factor). Second
# derivatives have larger finite-difference errors than first derivatives
DERIVATIVES = [
    pytest.param("d_polflux_dR", 1, 1.0, id="dR"),
    pytest.param("d_polflux_dZ", 1, 1.0, id="dZ"),
    pytest.param("d2_polflux_dR2", 1, 10.0, id="dR2"),
    pytest.param("d2_polflux_dZ2", 1, 10.0, id="dZ2"),
    pytest.param("d2_polflux_dRdZ", 2, 10.0, id="dRdZ"),
]


def make_fields(field_class):
    """The same field with exact and with finite-difference derivatives"""
    args = (B_T_AXIS, R_AXIS, MINOR_RADIUS_A, B_P_A)
    return (
        field_class(*args, derivative_method="analytic"),
        field_class(*args, derivative_method="numerical"),
    )


def sample_points(n=200, seed=0):
    """Random points with 0.1 <= rho <= 0.6, away from the magnetic axis,
    where the derivatives are singular"""
    rng = np.random.default_rng(seed)
    rho = rng.uniform(0.1, 0.6, n)
    theta = rng.uniform(-np.pi, np.pi, n)
    return R_AXIS + rho * np.cos(theta), rho * np.sin(theta)


def call(field, name, n_steps, R, Z, step):
    return getattr(field, name)(R, 0.0, Z, *([step] * n_steps))


##################################################
# Exact against numerical derivatives
##################################################


@pytest.mark.parametrize("field_class", FIELD_CLASSES)
@pytest.mark.parametrize(("name", "n_steps", "tol_factor"), DERIVATIVES)
def test_analytic_matches_numerical(field_class, name, n_steps, tol_factor):
    exact, numerical = make_fields(field_class)
    R, Z = sample_points()

    assert_allclose(
        call(numerical, name, n_steps, R, Z, DELTA),
        call(exact, name, n_steps, R, Z, DELTA),
        rtol=1e-6 * tol_factor,
        atol=1e-6 * tol_factor,
    )


@pytest.mark.parametrize(("name", "n_steps", "tol_factor"), DERIVATIVES)
def test_numerical_converges_at_second_order(name, n_steps, tol_factor):
    """Halving the step size should divide the finite-difference error by
    about 4, for steps large enough that rounding errors are negligible"""
    exact, numerical = make_fields(CircularCrossSectionField)
    R, Z = sample_points(seed=1)
    reference = call(exact, name, n_steps, R, Z, DELTA)

    errors = [
        np.abs(call(numerical, name, n_steps, R, Z, step) - reference).max()
        for step in (2e-3, 1e-3)
    ]

    assert 3.5 < errors[0] / errors[1] < 4.5


def test_analytic_known_values():
    """Exact derivatives at a point with rho = 0.5, theta = 45 degrees"""
    field, _ = make_fields(CircularCrossSectionField)
    offset = 0.5 / np.sqrt(2)
    R, Z = R_AXIS + offset, offset

    # psi = rho / a, so the gradient has magnitude 1 / a = 2
    assert_allclose(field.d_polflux_dR(R, 0.0, Z, DELTA), np.sqrt(2))
    assert_allclose(field.d_polflux_dZ(R, 0.0, Z, DELTA), np.sqrt(2))
    # Second derivatives are 1 / (2 a rho) or -1 / (2 a rho) = 2 or -2
    assert_allclose(field.d2_polflux_dR2(R, 0.0, Z, DELTA), 2.0)
    assert_allclose(field.d2_polflux_dZ2(R, 0.0, Z, DELTA), 2.0)
    assert_allclose(field.d2_polflux_dRdZ(R, 0.0, Z, DELTA, DELTA), -2.0)


def test_circular_field_is_consistent_with_poloidal_flux():
    """For CircularCrossSectionField, B_R = B_p_a dpsi/dZ / R and
    B_Z = -B_p_a dpsi/dR / R"""
    field, _ = make_fields(CircularCrossSectionField)
    R, Z = sample_points(seed=2)

    assert_allclose(
        field.B_R(R, 0.0, Z), B_P_A * field.d_polflux_dZ(R, 0.0, Z, DELTA) / R
    )
    assert_allclose(
        field.B_Z(R, 0.0, Z), -B_P_A * field.d_polflux_dR(R, 0.0, Z, DELTA) / R
    )


##################################################
# Options and construction
##################################################


@pytest.mark.parametrize("field_class", FIELD_CLASSES)
def test_derivative_method_defaults_to_analytic(field_class):
    field = field_class(B_T_AXIS, R_AXIS, MINOR_RADIUS_A, B_P_A)
    assert field.derivative_method == "analytic"


@pytest.mark.parametrize("field_class", FIELD_CLASSES)
def test_invalid_derivative_method(field_class):
    with pytest.raises(ValueError, match="derivative_method"):
        field_class(B_T_AXIS, R_AXIS, MINOR_RADIUS_A, B_P_A, derivative_method="spline")


@pytest.mark.parametrize(
    ("find_B_method", "field_class"),
    [
        ("analytical", CircularFluxSurfaceField),
        ("unit-tests", CircularCrossSectionField),
    ],
)
@pytest.mark.parametrize("derivative_method", ["analytic", "numerical"])
def test_create_magnetic_geometry_passes_derivative_method(
    find_B_method, field_class, derivative_method
):
    field = create_magnetic_geometry(
        geometry="cylindrical",
        find_B_method=find_B_method,
        interp_order_str="quintic",
        interp_order_int=5,
        interp_smoothing=0,
        magnetic_data_path=pathlib.Path("."),
        B_T_axis=B_T_AXIS,
        R_axis=R_AXIS,
        minor_radius_a=MINOR_RADIUS_A,
        B_p_a=B_P_A,
        derivative_method=derivative_method,
    )
    assert isinstance(field, field_class)
    assert field.derivative_method == derivative_method


@pytest.mark.parametrize(
    "base_class",
    [MagneticField_Cylindrical, MagneticField_Cartesian, CircularFieldBase],
)
def test_base_classes_cannot_be_instantiated(base_class):
    with pytest.raises(TypeError, match="abstract"):
        base_class()


def test_field_missing_derivatives_cannot_be_instantiated():
    """A field class has to implement every derivative, not just B and psi"""

    class IncompleteField(MagneticField_Cylindrical):
        def B_R(self, R, _, Z):
            return 0.0

        def B_T(self, R, _, Z):
            return 1.0

        def B_Z(self, R, _, Z):
            return 0.0

        def polflux(self, R, _, Z):
            return 0.0

    with pytest.raises(TypeError, match="d_polflux_dR"):
        IncompleteField()


##################################################
# Timing benchmark
##################################################


@pytest.mark.parametrize("field_class", FIELD_CLASSES)
def test_benchmark_derivative_timing(field_class, record_property):
    """Times exact against numerical derivatives. Only reports the timings
    (shown with ``pytest -s``, and in the JUnit XML), as timings depend on
    the machine"""
    exact, numerical = make_fields(field_class)
    n_points, repeats = 10_000, 5

    timings = {}
    for label, field in [("analytic", exact), ("numerical", numerical)]:
        best = np.inf
        for seed in range(repeats):
            # Fresh points each repeat, so the finite-difference cache in
            # `derivative` cannot be reused
            R, Z = sample_points(n_points, seed=100 + seed)
            start = time.perf_counter()
            for name, n_steps, _ in (p.values for p in DERIVATIVES):
                call(field, name, n_steps, R, Z, DELTA)
            best = min(best, time.perf_counter() - start)
        timings[label] = best
        record_property(f"{label}_seconds", best)

    speedup = timings["numerical"] / timings["analytic"]
    record_property("speedup", speedup)
    print(
        f"\n{field_class.__name__}: all 5 derivatives at {n_points} points, "
        f"best of {repeats}: analytic {timings['analytic'] * 1e3:.2f} ms, "
        f"numerical {timings['numerical'] * 1e3:.2f} ms ({speedup:.1f}x)"
    )

    assert np.isfinite(speedup)


def test_circular_flux_surface_field():
    """The field of beam-tracing/Scotty#161, as find_B_method="analytical" gives
    in v3: psi_pol = (r/a)^2, B_p linear in r at fixed R, divergence-free and
    tangent to the flux surfaces. Ported from #161's test_circular_flux_surface"""
    B_T_axis, R_axis, a, B_p_a = 1.0, 2.0, 1.0, 0.5
    field = CircularFluxSurfaceField(B_T_axis, R_axis, a, B_p_a)

    # psi_pol = (r/a)^2, so rho_pol = r/a
    assert_allclose(field.polflux(R_axis + 0.5, 0.0, 0.0), 0.25)
    assert_allclose(field.polflux(R_axis, 0.0, -a), 1.0)

    # B_p_a is the poloidal field on the LCFS at R = R_axis, and B_p is linear in r at fixed R
    assert np.isclose(field.B_R(R_axis, 0.0, a), B_p_a), "top"
    assert np.isclose(field.B_R(R_axis, 0.0, 0.5 * a), 0.5 * B_p_a), "half"
    assert np.isclose(field.magnitude(R_axis, 0.0, 0.0), B_T_axis), "on axis"
    assert_allclose(field.unitvector(R_axis, 0.0, 0.0), [0, 1, 0], atol=1e-15)

    # B_R = (1/R) dPsi/dZ and B_Z = -(1/R) dPsi/dR with Psi = B_p_a a R_axis psi_pol / 2
    width = a + 0.05
    R = np.linspace(R_axis - width, R_axis + width, 99)
    Z = np.linspace(-width, width, 101)
    R_grid, Z_grid = np.meshgrid(R, Z, indexing="ij")
    B_R = field.B_R(R_grid, 0.0, Z_grid)
    B_Z = field.B_Z(R_grid, 0.0, Z_grid)
    Psi = 0.5 * B_p_a * a * R_axis * field.polflux(R_grid, 0.0, Z_grid)
    dPsi_dR, dPsi_dZ = np.gradient(Psi, R, Z, edge_order=2)
    assert_allclose(B_R, dPsi_dZ / R_grid, rtol=1e-6, atol=1e-6)
    assert_allclose(B_Z, -dPsi_dR / R_grid, rtol=1e-6, atol=1e-6)

    # Divergence-free: (1/R) d(R B_R)/dR + dB_Z/dZ = 0
    divergence = np.gradient(
        R_grid * B_R, R, axis=0, edge_order=2
    ) / R_grid + np.gradient(B_Z, Z, axis=1, edge_order=2)
    assert_allclose(divergence, 0.0, atol=1e-10)

    # Tangent to the flux surfaces: B . grad(psi) = 0, with the exact derivatives
    tangency = B_R * field.d_polflux_dR(
        R_grid, 0.0, Z_grid, DELTA
    ) + B_Z * field.d_polflux_dZ(R_grid, 0.0, Z_grid, DELTA)
    assert_allclose(tangency, 0.0, atol=1e-14)

    # Same direction of rotation as CircularCrossSectionField
    total_sign = np.sign(B_R) * np.sign(B_Z)
    assert total_sign[0, 0] == -1, "Top left"
    assert total_sign[0, -1] == 1, "Top right"
    assert total_sign[-1, 0] == 1, "Bottom left"
    assert total_sign[-1, -1] == -1, "Bottom right"


def test_circular_flux_surface_derivatives_on_axis():
    """psi_pol = (r/a)^2 is smooth on the magnetic axis, so its exact derivatives
    are finite there (the r/a fields' are singular)"""
    field = CircularFluxSurfaceField(B_T_AXIS, R_AXIS, MINOR_RADIUS_A, B_P_A)
    a2 = MINOR_RADIUS_A**2
    assert field.d_polflux_dR(R_AXIS, 0.0, 0.0, DELTA) == 0.0
    assert field.d_polflux_dZ(R_AXIS, 0.0, 0.0, DELTA) == 0.0
    assert np.isclose(field.d2_polflux_dR2(R_AXIS, 0.0, 0.0, DELTA), 2 / a2)
    assert np.isclose(field.d2_polflux_dZ2(R_AXIS, 0.0, 0.0, DELTA), 2 / a2)
    assert field.d2_polflux_dRdZ(R_AXIS, 0.0, 0.0, DELTA, DELTA) == 0.0
