from scotty.derivatives import derivative

import numpy as np


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
