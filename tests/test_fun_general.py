from scotty.fun_general import (
    freq_GHz_to_wavenumber,
    find_nearest,
    read_floats_into_list_until,
    find_Psi_3D_lab,
    find_Psi_3D_lab_Cartesian,
    make_array_3x3,
    K_magnitude
)

import io
from textwrap import dedent

from scipy.constants import c as speed_of_light
import numpy as np
import numpy.testing as npt


def test_freq_GHz_to_wavenumber():
    assert np.isclose(freq_GHz_to_wavenumber(1.0 / (2 * np.pi)), 1e9 / speed_of_light)


def test_find_nearest():
    data = [1, 3, 5, 7, 9]
    assert find_nearest(data, 5.5) == 2
    assert find_nearest(data, 6.5) == 3
    assert find_nearest(data, 0.5) == 0
    assert find_nearest(data, 10) == 4


def test_read_floats_into_list():
    data = io.StringIO(
        dedent(
            """
            5.00000000e-01
            1.00000000e+00
            1.50000000e+00
            Grid: Z-coordinates
            -0.55   -0.42777778   -0.30555556   -0.18333333   -0.061111111
            0.061111111   0.18333333   0.30555556   0.42777778   0.55
            Magnetic field: B_R
            1 2 3 4
            """
        )
    )
    x_coords = read_floats_into_list_until("Z-coordinates", data)
    z_coords = read_floats_into_list_until("B_R", data)
    B_r = read_floats_into_list_until("not in file", data)

    expected_x = [0.5, 1.0, 1.5]
    expected_z = [
        -0.55,
        -0.42777778,
        -0.30555556,
        -0.18333333,
        -0.061111111,
        0.061111111,
        0.18333333,
        0.30555556,
        0.42777778,
        0.55,
    ]
    expected_B_r = [1, 2, 3, 4]

    npt.assert_allclose(x_coords, expected_x)
    npt.assert_allclose(z_coords, expected_z)
    npt.assert_allclose(B_r, expected_B_r)


def random_symmetric_matrix():
    array = np.random.random((3, 3))
    array[0, 1] = array[1, 0]
    array[0, 2] = array[2, 0]
    array[1, 2] = array[2, 1]
    return array


def test_find_Psi_3D_lab_symmetry():
    Psi_lab_Cartesian = random_symmetric_matrix()

    Psi_lab_cylindrical = find_Psi_3D_lab(Psi_lab_Cartesian, 2, np.pi, 42, 64)

    # result should be symmetric
    assert Psi_lab_cylindrical[0, 1] == Psi_lab_cylindrical[1, 0]
    assert Psi_lab_cylindrical[0, 2] == Psi_lab_cylindrical[2, 0]
    assert Psi_lab_cylindrical[1, 2] == Psi_lab_cylindrical[2, 1]

    # ZZ component unchanged
    assert Psi_lab_cylindrical[2, 2].real == Psi_lab_Cartesian[2, 2]


def test_convert_Psi_roundtrip():
    Psi_original = random_symmetric_matrix()
    q_R = 2
    q_zeta = np.pi
    K_R = 42
    K_zeta = 64
    Psi_cylindrical = find_Psi_3D_lab(Psi_original, q_R, q_zeta, K_R, K_zeta)
    Psi_cartesian = find_Psi_3D_lab_Cartesian(Psi_cylindrical, q_R, q_zeta, K_R, K_zeta)

    npt.assert_allclose(Psi_cartesian.real, Psi_original)
    npt.assert_allclose(Psi_cartesian.imag, np.zeros_like(Psi_original))


def test_Psi_cylindrical():
    xx = 1
    yy = 2
    zz = 3
    xy = 4
    xz = 5
    yz = 6
    Psi_lab_Cartesian = np.array(
        [
            [xx, xy, xz],
            [xy, yy, yz],
            [xz, yz, zz],
        ]
    )

    R = 2
    Psi_lab_cylindrical = find_Psi_3D_lab(Psi_lab_Cartesian, R, np.pi, 0, 0)
    expected = np.array(
        [
            [xx, xy * R, -xz],
            [xy * R, yy * R**2, -yz * R],
            [-xz, -yz * R, zz],
        ]
    )
    npt.assert_allclose(Psi_lab_cylindrical, expected + 0j)


def test_make_array_3x3():
    A = np.array([[1, 2], [3, 4]])
    B = make_array_3x3(A)

    expected = np.array([[1, 2, 0], [3, 4, 0], [0, 0, 0]])
    npt.assert_array_equal(B, expected)


def test_K_magnitude():
    # Pythagorean quadruple (2, 10, 11, 15)
    K_R = 2
    K_zeta = 20
    K_Z = 11
    q_R = 2
    expected_K = 15
    assert K_magnitude(K_R, K_zeta, K_Z, q_R) == expected_K
