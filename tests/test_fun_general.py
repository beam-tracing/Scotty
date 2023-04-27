from scotty.fun_general import (
    freq_GHz_to_wavenumber,
    find_nearest,
    read_floats_into_list_until,
    find_Psi_3D_lab,
    find_Psi_3D_lab_Cartesian,
    make_array_3x3,
    K_magnitude,
    contract_special,
    find_Psi_3D_plasma_discontinuous,
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


def test_contract_special_vector_vector():
    vector1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    vector2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    expected = np.array([1, 1, 1])
    result = contract_special(vector1, vector2)
    npt.assert_array_equal(result, expected)


def test_contract_special_matrix_vector():
    matrix = np.array(
        [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        ]
    )
    vector = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    result = contract_special(matrix, vector)
    npt.assert_array_equal(result, expected)


def test_contract_special_vector_matrix():
    matrix = np.array(
        [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        ]
    )
    vector = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    result = contract_special(vector, matrix)
    npt.assert_array_equal(result, expected)


def test_K_magnitude():
    # Pythagorean quadruple (2, 10, 11, 15)
    K_R = 2
    K_zeta = 20
    K_Z = 11
    q_R = 2
    expected_K = 15
    assert K_magnitude(K_R, K_zeta, K_Z, q_R) == expected_K


def test_find_Psi_3D_plasma_discontinuous():
    ## Values from a test case that I used
    ## TODO: Integrate this test with a circular-flux-surface case which has discontinuous ne
    ## Find values from a simple slab case and test them

    Psi_vacuum_3D = np.array(
        [
            [
                22.96480274 + 25.35603703j,
                -288.34856095 - 200.18496508j,
                -76.01536319 - 83.93054303j,
            ],
            [
                -288.34856095 - 200.18496508j,
                3901.7586168 + 2724.12830862j,
                -22.50617362 - 24.84965268j,
            ],
            [
                -76.01536319 - 83.93054303j,
                -22.50617362 - 24.84965268j,
                625.89678142 + 691.06894368j,
            ],
        ]
    )
    K_v_R = -720.1663330442489
    K_v_zeta = -213.22253614625006
    K_v_Z = -89.39673984646942
    K_p_R = -572.3610161829057
    K_p_zeta = -213.22253614625006
    K_p_Z = -116.25073909098793
    dH_dKR = -0.0021297263342362482
    dH_dKzeta = -0.00018858310699032543
    dH_dKZ = -0.00044100184796613817
    dH_dR = -2.2863224601654686
    dH_dZ = 0.40208647341111176
    dpolflux_dR = 1.9677859515843108
    dpolflux_dZ = -0.3575170608158262
    d2polflux_dR2 = 0.12782996883231587
    d2polflux_dZ2 = 3.8721770234673154
    d2polflux_dRdZ = 0.7035172444602721

    Psi_3D_plasma = find_Psi_3D_plasma_discontinuous(
        Psi_vacuum_3D,
        K_v_R,
        K_v_zeta,
        K_v_Z,
        K_p_R,
        K_p_zeta,
        K_p_Z,
        dH_dKR,  # In the plasma
        dH_dKzeta,  # In the plasma
        dH_dKZ,  # In the plasma
        dH_dR,  # In the plasma
        dH_dZ,  # In the plasma
        dpolflux_dR,  # Continuous
        dpolflux_dZ,  # Continuous
        d2polflux_dR2,  # Continuous
        d2polflux_dZ2,  # Continuous
        d2polflux_dRdZ,  # Continuous
    )

    expected_Psi_3D_plasma = np.array(
        [
            [
                -1035.11039468 + 51.2582456j,
                -458.03354529 - 237.47347794j,
                10.33282589 - 145.99156328j,
            ],
            [
                -458.03354529 - 237.47347794j,
                3901.7586168 + 2724.12830862j,
                543.49055857 - 18.07489168j,
            ],
            [
                10.33282589 - 145.99156328j,
                543.49055857 - 18.07489168j,
                629.44689491 + 712.76503166j,
            ],
        ]
    )

    npt.assert_allclose(Psi_3D_plasma, expected_Psi_3D_plasma)
