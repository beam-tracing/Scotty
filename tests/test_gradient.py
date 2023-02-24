from scotty.fun_CFD import cfd_gradient


import numpy as np


def myfun(x, y):
    return y**2 * (1 + np.sin(x))


def test_gradient():
    grad_x = cfd_gradient(myfun, x=0, y=3, directions="x", dx=1e-4)
    assert np.isclose(grad_x, 9)

    grad_y = cfd_gradient(myfun, x=0, y=3, directions="y", dx=1e-4)
    assert np.isclose(grad_y, 6)


def test_second_gradient():
    grad_x = cfd_gradient(myfun, x=0, y=3, directions="x", dx=1e-4, d_order=2)
    assert np.isclose(grad_x, 0, atol=1e-6)

    grad_y = cfd_gradient(myfun, x=0, y=3, directions="y", dx=1e-4, d_order=2)
    assert np.isclose(grad_y, 2)


def test_gradient_xy():
    grad_xy = cfd_gradient(
        myfun, x=0, y=3, directions=("x", "y"), dx=(1e-4, 1e-4), d_order=2
    )
    assert np.isclose(grad_xy, 6)

    # Shouldn't depend on order of directions
    grad_yx = cfd_gradient(
        myfun, x=0, y=3, directions=("y", "x"), dx=(1e-4, 1e-4), d_order=2
    )
    assert np.isclose(grad_yx, 6)
