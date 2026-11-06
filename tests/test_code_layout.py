"""The modules moved into `scotty.antenna` and `scotty.diagnostics` can still
be imported from where they were, and give the same objects"""

import numpy as np

from scotty.antenna import gaussian, horns, lenses
from scotty.diagnostics import dbs


def test_old_module_names_are_the_new_modules():
    import scotty.hornpy
    import scotty.lensalot

    assert scotty.hornpy is horns
    assert scotty.lensalot is lenses
    from scotty.hornpy import make_my_horn
    from scotty.lensalot import make_my_lens

    assert make_my_horn is horns.make_my_horn
    assert make_my_lens is lenses.make_my_lens


def test_gaussian_functions_from_fun_general():
    from scotty import fun_general
    from scotty.fun_general import find_waist

    assert find_waist is gaussian.find_waist
    for name in fun_general._MOVED_TO_ANTENNA_GAUSSIAN:
        assert getattr(fun_general, name) is getattr(gaussian, name)


def test_dbs_analysis_from_analysis():
    from scotty.analysis import dbs_analysis

    assert dbs_analysis is dbs.dbs_analysis


def test_circular_gaussian_Psi_w():
    """K0/R + 2i/W^2 on the diagonal, as `scotty.launch` had inline"""
    K0, width, curvature = 1150.0, 0.04, -0.25
    np.testing.assert_array_equal(
        gaussian.circular_gaussian_Psi_w(K0, width, curvature),
        (K0 * curvature + 2j / width**2) * np.eye(2),
    )
