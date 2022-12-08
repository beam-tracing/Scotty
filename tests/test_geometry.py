from scotty import geometry

import numpy as np
import numpy.testing as npt


def test_circular():
    B_T_axis = 1.0
    R_axis = 2.0
    minor_radius_a = 1.0
    B_p_a = 0.5
    field = geometry.CircularCrossSectionField(
        B_T_axis=B_T_axis, R_axis=R_axis, minor_radius_a=minor_radius_a, B_p_a=B_p_a
    )

    assert np.isclose(field.B_T(R_axis, 0.0), B_T_axis), "B_T on axis, scalar"
    npt.assert_allclose(
        field.B_T([R_axis, R_axis], [-1, 1]), [B_T_axis, B_T_axis]
    ), "B_T, array"

    # Include buffer for gradient near edge
    width = minor_radius_a + 0.05
    # Different grid sizes to capture transpose errors
    R = np.linspace(R_axis - width, R_axis + width, 99)
    Z = np.linspace(-width, width, 101)
    R_grid, Z_grid = np.meshgrid(R, Z, indexing="ij")
    B_R = field.B_R(R_grid, Z_grid)
    B_Z = field.B_Z(R_grid, Z_grid)

    psi = field.poloidal_flux(R_grid, Z_grid)
    grad_psi = np.gradient(psi, R, Z)
    calculated_B_R = grad_psi[1] * B_p_a / R_grid
    calculated_B_Z = -grad_psi[0] * B_p_a / R_grid

    mask = (psi > 0.1) & (psi <= 1)

    npt.assert_allclose(B_R[mask], calculated_B_R[mask], 2e-3, 2e-3)
    npt.assert_allclose(B_Z[mask], calculated_B_Z[mask], 2e-3, 2e-3)

    # Check that poloidal field rotates in the correct direction
    total_sign = np.sign(B_R) * np.sign(B_Z)
    assert total_sign[0, 0] == -1, "Top left"
    assert total_sign[0, -1] == 1, "Top right"
    assert total_sign[-1, 0] == 1, "Bottom left"
    assert total_sign[-1, -1] == -1, "Bottom right"


def test_interpolated():
    B_T_axis = 1.0
    R_axis = 2.0
    minor_radius_a = 1.0
    B_p_a = 0.5
    circular_field = geometry.CircularCrossSectionField(
        B_T_axis=B_T_axis, R_axis=R_axis, minor_radius_a=minor_radius_a, B_p_a=B_p_a
    )

    R = np.linspace(R_axis - minor_radius_a, R_axis + minor_radius_a)
    Z = np.linspace(-minor_radius_a, minor_radius_a)
    R_grid, Z_grid = np.meshgrid(R, Z, indexing="ij")
    B_R = circular_field.B_R(R_grid, Z_grid)
    B_T = circular_field.B_T(R_grid, Z_grid)
    B_Z = circular_field.B_Z(R_grid, Z_grid)
    psi = circular_field.poloidal_flux(R_grid, Z_grid)

    field = geometry.InterpolatedField(R, Z, B_R, B_T, B_Z, psi)

    assert np.isclose(field.B_T(R_axis, 0.0), B_T_axis), "B_T on axis, scalar"
    npt.assert_allclose(
        field.B_T([R_axis, R_axis], [-1, 1]), [B_T_axis, B_T_axis], rtol=5e-6
    ), "B_T, array"

    # Not quite from the axis, as interpolation particularly bad there
    R_midplane = np.linspace(R_axis + 0.1, R_axis + minor_radius_a, 10)
    Z_vertical = np.linspace(0.1, minor_radius_a, 10)

    npt.assert_allclose(
        field.B_R(R_axis, Z_vertical), circular_field.B_R(R_axis, Z_vertical), rtol=1e-2
    )
    npt.assert_allclose(
        field.B_T(R_midplane, 0.0), circular_field.B_T(R_midplane, 0.0), rtol=1e-2
    )
    npt.assert_allclose(
        field.B_Z(R_midplane, 0.0), circular_field.B_Z(R_midplane, 0.0), rtol=1e-2
    )
    npt.assert_allclose(
        field.poloidal_flux(R_midplane, 0.0),
        circular_field.poloidal_flux(R_midplane, 0.0),
        rtol=1e-3,
    )
