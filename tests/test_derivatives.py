from scotty.derivatives import derivative, cache

import numpy as np
import numpy.testing as npt
from time import sleep, time_ns


def test_first_derivative():
    result = derivative(
        lambda q_R: np.sin(q_R),
        "q_R",
        {"q_R": 0.0},
        spacings={"q_R": 1e-8},
        stencil="d1_CFD2",
    )

    assert np.isclose(result, 1.0)


def test_2D_derivative():
    cache = {}
    result = derivative(
        lambda q_R, q_Z: np.sin(q_R) * q_Z**2,
        ("q_R", "q_Z"),
        {"q_R": 0.0, "q_Z": 2.0},
        spacings={"q_R": 1e-8, "q_Z": 1e-8},
        stencil="d2d2_CFD_CFD2",
        cache=cache,
    )
    assert np.isclose(result, 4.0)


@cache
def add_points(x, y, z):
    sleep(0.5)
    return x + y + z


def test_cache_free_function():
    start = time_ns()
    result1 = add_points(np.array([1, 2, 3]), np.array([1, 2, 3]), 2.1)
    end = time_ns()
    time1 = end - start

    start = time_ns()
    result2 = add_points(np.array([1, 2, 3]), np.array([1, 2, 3]), 2.1)
    end = time_ns()
    time2 = end - start

    expected = np.array([4.1, 6.1, 8.1])

    npt.assert_array_equal(result1, expected)
    npt.assert_array_equal(result2, expected)
    # First call should be slow, second much faster
    assert time1 >= 5e8
    assert time2 < 1e6


class ObjectWithMethod:
    def __init__(self, x):
        self.x = x

    @cache
    def add_points(self, y, z):
        sleep(0.5)
        return self.x + y + z


def test_cache_method():
    foo = ObjectWithMethod(np.array([2, 3, 4]))

    start = time_ns()
    result1 = foo.add_points(np.array([1, 2, 3]), 2.1)
    end = time_ns()
    time1 = end - start

    start = time_ns()
    result2 = foo.add_points(np.array([1, 2, 3]), 2.1)
    end = time_ns()
    time2 = end - start

    expected = np.array([5.1, 7.1, 9.1])

    npt.assert_array_equal(result1, expected)
    npt.assert_array_equal(result2, expected)
    # First call should be slow, second much faster
    assert time1 >= 5e8
    assert time2 < 1e6


class Functor:
    def __init__(self, x):
        self.x = x

    @cache
    def __call__(self, y, z):
        sleep(0.5)
        return self.x + y + z


def test_cache_functor():
    foo = Functor(np.array([2, 3, 4]))

    start = time_ns()
    result1 = foo(np.array([1, 2, 3]), 2.1)
    end = time_ns()
    time1 = end - start

    start = time_ns()
    result2 = foo(np.array([1, 2, 3]), 2.1)
    end = time_ns()
    time2 = end - start

    expected = np.array([5.1, 7.1, 9.1])

    npt.assert_array_equal(result1, expected)
    npt.assert_array_equal(result2, expected)
    # First call should be slow, second much faster
    assert time1 >= 5e8
    assert time2 < 1e6
