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

    R = np.linspace(R_axis - minor_radius_a, R_axis + minor_radius_a, 10)
    Z = np.linspace(-minor_radius_a / 2, minor_radius_a / 2, 10)
    R_grid, Z_grid = np.meshgrid(R, Z, indexing="ij")
    B_R = field.B_R(R_grid, Z_grid)
    B_Z = field.B_Z(R_grid, Z_grid)
    calculated_B_poloidal = -np.sqrt(B_R**2 + B_Z**2)
    npt.assert_allclose(calculated_B_poloidal, field.B_p(R_grid, Z_grid))

    # Check that poloidal field rotates in the correct direction
    total_sign = np.sign(B_R) * np.sign(B_Z)
    assert total_sign[0, 0] == -1, "Top left"
    assert total_sign[0, -1] == 1, "Top right"
    assert total_sign[-1, 0] == 1, "Bottom left"
    assert total_sign[-1, -1] == -1, "Bottom right"
