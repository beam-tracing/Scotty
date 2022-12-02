from scotty import geometry

import numpy as np
import numpy.testing as npt


def test_circular():
    B_T_axis = 1.0
    R_axis = 2.0
    minor_radius_a = 1.0
    B_p_a = 0.5
    field = geometry.CircularCrossSection(
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
