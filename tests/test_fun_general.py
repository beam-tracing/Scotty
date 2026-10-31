import io
import numpy as np
import numpy.testing as npt
import pytest
from scipy.constants import c as speed_of_light
from scotty.fun_general import (
    dot,
    find_K_labframe_cyl_to_cart,
    freq_GHz_to_wavenumber,
    find_K_magnitude,
    find_nearest,
    find_Psi_3D_labframe_cart_to_cyl,
    find_Psi_3D_labframe_cyl_to_cart,
    find_q_labframe_cart_to_cyl,
    find_q_labframe_cyl_to_cart,
    make_array_3x3,
    read_floats_into_list_until,
)
from scotty.typing import ComplexFloatArray
from textwrap import dedent
from typing import cast


def test_freq_GHz_to_wavenumber():
    assert np.isclose(freq_GHz_to_wavenumber(1.0 / (2 * np.pi)), 1e9 / speed_of_light)


def test_find_nearest():
    data = np.array([1, 3, 5, 7, 9])
    assert find_nearest(data, 5.5) == 2
    assert find_nearest(data, 6.5) == 3
    assert find_nearest(data, 0.5) == 0
    assert find_nearest(data, 10) == 4


def test_read_floats_into_list():
    data = io.StringIO(dedent("""
            5.00000000e-01
            1.00000000e+00
            1.50000000e+00
            Grid: Z-coordinates
            -0.55   -0.42777778   -0.30555556   -0.18333333   -0.061111111
            0.061111111   0.18333333   0.30555556   0.42777778   0.55
            Magnetic field: B_R
            1 2 3 4
            """))
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


def random_symmetric_matrix() -> ComplexFloatArray:
    array = np.random.random((3, 3))
    array[0, 1] = array[1, 0]
    array[0, 2] = array[2, 0]
    array[1, 2] = array[2, 1]
    return cast(ComplexFloatArray, array)


def test_find_Psi_3D_lab_symmetry():
    Psi_lab_Cartesian = random_symmetric_matrix()

    Psi_lab_cylindrical = find_Psi_3D_labframe_cart_to_cyl(
        Psi_lab_Cartesian, np.array([42, 64, 0]), np.array([2, np.pi, 0.123])
    )

    # result should be symmetric
    assert Psi_lab_cylindrical[0, 1] == Psi_lab_cylindrical[1, 0]
    assert Psi_lab_cylindrical[0, 2] == Psi_lab_cylindrical[2, 0]
    assert Psi_lab_cylindrical[1, 2] == Psi_lab_cylindrical[2, 1]

    # ZZ component unchanged
    assert Psi_lab_cylindrical[2, 2].real == Psi_lab_Cartesian[2, 2]


def test_convert_Psi_roundtrip():
    q_cyld = np.array([2, np.pi, 0.123])
    q_cart = find_q_labframe_cyl_to_cart(q_cyld)
    K_cyld = np.array([42, 64, 0])
    K_cart = find_K_labframe_cyl_to_cart(K_cyld, q_cyld)
    Psi_original_cyld = random_symmetric_matrix()

    Psi_cart = find_Psi_3D_labframe_cyl_to_cart(
        Psi_original_cyld,
        K_cyld,
        q_cyld,
    )

    Psi_cyld = find_Psi_3D_labframe_cart_to_cyl(
        Psi_cart,
        K_cart,
        q_cart,
    )

    npt.assert_allclose(Psi_cyld.real, Psi_original_cyld)
    npt.assert_allclose(Psi_cyld.imag, np.zeros_like(Psi_original_cyld))


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

    q_cyld = np.array([2, np.pi, 0.123])
    q_cart = find_q_labframe_cyl_to_cart(q_cyld)
    K_cyld = K_cart = np.array([0, 0, 0])
    Psi_cyld = find_Psi_3D_labframe_cart_to_cyl(
        Psi_lab_Cartesian,
        K_cart,
        q_cart,
    )

    R = q_cyld[0]
    expected = np.array(
        [
            [xx, xy * R, -xz],
            [xy * R, yy * R**2, -yz * R],
            [-xz, -yz * R, zz],
        ]
    )
    npt.assert_allclose(Psi_cyld, expected + 0j)


def _quadratic_phase_case(seed):
    """Exact test field s(x) = k.x + x.A.x / 2 with complex symmetric A: the
    Cartesian Psi is A everywhere and K_cart = k + A x. The cylindrical Psi
    (dimensionless K_zeta = ds/dzeta) is the Hessian of
    s(R cos zeta, R sin zeta, Z), found here by central differences, so the
    coordinate-curvature terms (K_zeta / R in Psi_Rzeta, -R K_R in
    Psi_zetazeta) are tested rather than assumed."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    A = 50 * (A + A.T)
    k = 500 * rng.normal(size=3)
    q_cyl = np.array(
        [rng.uniform(0.5, 2.5), rng.uniform(-np.pi, np.pi), rng.uniform(-1, 1)]
    )
    q_cart = find_q_labframe_cyl_to_cart(q_cyl)

    def s(q):
        x = np.array([q[0] * np.cos(q[1]), q[0] * np.sin(q[1]), q[2]])
        return k @ x + 0.5 * x @ A @ x

    h = 1e-4
    Psi_cyl = np.empty((3, 3), dtype=complex)
    for i in range(3):
        for j in range(3):
            ei, ej = h * np.eye(3)[i], h * np.eye(3)[j]
            Psi_cyl[i, j] = (
                s(q_cyl + ei + ej)
                - s(q_cyl + ei - ej)
                - s(q_cyl - ei + ej)
                + s(q_cyl - ei - ej)
            ) / (4 * h**2)

    K_cart = k + A @ q_cart
    K_cyl = np.array(
        [
            K_cart[0] * np.cos(q_cyl[1]) + K_cart[1] * np.sin(q_cyl[1]),
            q_cyl[0] * (-K_cart[0] * np.sin(q_cyl[1]) + K_cart[1] * np.cos(q_cyl[1])),
            K_cart[2],
        ]
    )
    return A, Psi_cyl, K_cart, q_cart, K_cyl, q_cyl


@pytest.mark.parametrize("seed", range(5))
def test_Psi_conversion_exact_field(seed):
    A, Psi_cyl, K_cart, q_cart, K_cyl, q_cyl = _quadratic_phase_case(seed)
    npt.assert_allclose(
        find_Psi_3D_labframe_cart_to_cyl(A, K_cart, q_cart),
        Psi_cyl,
        atol=1e-6 * np.max(np.abs(Psi_cyl)),
    )
    npt.assert_allclose(
        find_Psi_3D_labframe_cyl_to_cart(Psi_cyl, K_cyl, q_cyl),
        A,
        atol=1e-6 * np.max(np.abs(A)),
    )


def test_Psi_conversion_exact_field_stacked():
    cases = [_quadratic_phase_case(seed) for seed in range(4)]
    A, Psi_cyl, K_cart, q_cart, K_cyl, q_cyl = (
        np.array([case[i] for case in cases]) for i in range(6)
    )
    npt.assert_allclose(
        find_Psi_3D_labframe_cart_to_cyl(A, K_cart.T, q_cart.T),
        Psi_cyl,
        atol=1e-6 * np.max(np.abs(Psi_cyl)),
    )
    npt.assert_allclose(
        find_Psi_3D_labframe_cyl_to_cart(Psi_cyl, K_cyl.T, q_cyl.T),
        A,
        atol=1e-6 * np.max(np.abs(A)),
    )


def test_make_array_3x3():
    A = np.array([[1, 2], [3, 4]])
    B = make_array_3x3(A)

    expected = np.array([[1, 2, 0], [3, 4, 0], [0, 0, 0]])
    npt.assert_array_equal(B, expected)


def test_dot_vector_vector():
    vector1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    vector2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    expected = np.array([1, 1, 1])
    result = dot(vector1, vector2)
    npt.assert_array_equal(result, expected)


def test_dot_matrix_vector():
    matrix = np.array(
        [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        ]
    )
    vector = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    result = dot(matrix, vector)
    npt.assert_array_equal(result, expected)


def test_dot_vector_matrix():
    matrix = np.array(
        [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        ]
    )
    vector = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    result = dot(vector, matrix)
    npt.assert_array_equal(result, expected)


def test_K_magnitude():
    # Pythagorean quadruple (2, 10, 11, 15)
    K_R = 2
    K_zeta = 20
    K_Z = 11
    q_R = 2
    expected_K = 15
    assert find_K_magnitude(False, K_R, K_zeta, K_Z, q_R) == expected_K


@pytest.mark.parametrize(
    ("cartesian", "expected_cylindrical"),
    (
        ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        ((0.0, 1.0, 0.0), (1.0, np.pi / 2, 0.0)),
        ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
        ((np.sqrt(2) / 2, np.sqrt(2) / 2, 2.2), (1.0, np.pi / 4, 2.2)),
    ),
)
def test_cartesian_to_cylindrical(cartesian, expected_cylindrical):
    cylindrical = find_q_labframe_cart_to_cyl(np.array(cartesian))
    npt.assert_allclose(cylindrical, expected_cylindrical)


@pytest.mark.parametrize(
    ("cylindrical", "expected_cartesian"),
    (
        ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        ((1.0, np.pi / 2, 0.0), (0.0, 1.0, 0.0)),
        ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
        ((1.0, np.pi / 4, 2.2), (np.sqrt(2) / 2, np.sqrt(2) / 2, 2.2)),
    ),
)
def test_cylindrical_to_cylindrical(cylindrical, expected_cartesian):
    cartesian = find_q_labframe_cyl_to_cart(np.array(cylindrical))
    npt.assert_allclose(cartesian, expected_cartesian, atol=1e-15)
