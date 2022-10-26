from scotty.lensalot import make_my_lens

import numpy as np
import numpy.testing as nt
from scipy.constants import c as speed_of_light

import pytest


@pytest.mark.parametrize("name", ["MAST_V_band", "MAST_Q_band", "DBS_UCLA_MAST-U"])
def test_thin_lens(name):
    lens = make_my_lens(name, "thin")

    psi_in = np.array([[1 + 0j, 0 + 0j], [0 + 0j, 1 + 0j]])

    frequency = speed_of_light / (2 * np.pi * 1e9)

    psi = lens.output_beam(psi_in, frequency)

    real_diagonal = 1 - (1.0 / lens.focal_length)
    psi_expected = np.array(
        [[real_diagonal + 0j, 0 + 0j], [0 + 0j, real_diagonal + 0j]]
    )
    nt.assert_allclose(psi, psi_expected)


@pytest.mark.parametrize("name", ["MAST_V_band", "MAST_Q_band"])
def test_thick_lens(name):
    lens = make_my_lens(name, "thick")

    psi_in = np.array([[1 + 0j, 0 + 0j], [0 + 0j, 1 + 0j]])

    frequency = speed_of_light / (2 * np.pi * 1e9)

    psi = lens.output_beam(psi_in, frequency)

    real_diagonal = 1 - (1.0 / lens.focal_length)
    psi_expected = np.array(
        [[real_diagonal + 0j, 0 + 0j], [0 + 0j, real_diagonal + 0j]]
    )
    # Very high tolerance as expected result should be approximate
    # thin solution
    nt.assert_allclose(psi, psi_expected, rtol=1e-2)


@pytest.mark.parametrize(
    ("name", "real_diagonal"), [("MAST_V_band", -5.056606), ("MAST_Q_band", -2.445143)]
)
def test_hyperbolic_lens(name, real_diagonal):
    """Golden answer test, expected result may need updating"""

    lens = make_my_lens(name, "hyperbolic")

    psi_in = np.array([[1 + 0j, 0 + 0j], [0 + 0j, 1 + 0j]])

    frequency = speed_of_light / (2 * np.pi * 1e9)

    psi = lens.output_beam(psi_in, frequency)

    psi_expected = np.array(
        [[real_diagonal + 0j, 0 + 0j], [0 + 0j, real_diagonal + 0j]]
    )
    nt.assert_allclose(psi, psi_expected)
