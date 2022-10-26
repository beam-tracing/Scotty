from scotty.hornpy import make_my_horn

import numpy as np
import numpy.testing as nt


def test_MAST_V_band():
    horn = make_my_horn("MAST_V_band")

    # Doesn't depend on frequency, so should be fine at infinity
    width, curvature = horn.output_beam(np.inf)

    expected_width = 0.0038551918335419165

    assert np.isclose(width, expected_width)
    assert np.isclose(width / 0.644, horn.aperture_radius)
    assert curvature == 0


def test_MAST_Q_band():
    horn = make_my_horn("MAST_Q_band")
    horn.width_method = "Goldsmith_asymmetric"

    width, curvature = horn.output_beam()

    expected_width = [0.01609344, 0.01170432]
    expected_curvature = 14.152397479359188

    nt.assert_allclose(width, expected_width)
    nt.assert_allclose(curvature, expected_curvature)
