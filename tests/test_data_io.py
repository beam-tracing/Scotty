import numpy as np
from numpy.testing import assert_allclose

from scotty.beam_me_up import beam_me_up
from scotty.fun_general import find_q_labframe_cyl_to_cart
from scotty.init_bruv import get_parameters_for_Scotty


def test_initial_position_is_the_start_of_the_ray(tmp_path):
    """q_initial is the first point in the plasma, where the solver starts;
    not q_launch (the vacuum launch position) and, by up to a small nudge
    inwards, not the plasma boundary ("entry") either. In both coordinate
    systems"""
    kwargs = get_parameters_for_Scotty("DBS_synthetic")
    kwargs.update(len_tau=52, interp_smoothing=0)
    so = beam_me_up(**kwargs, output_path=tmp_path)["solver_output"]

    first = [so["q_R_output"][0], so["q_zeta_output"][0], so["q_Z_output"][0]]
    assert_allclose(so["q_initial_cylindrical"], first)
    assert_allclose(so["q_initial_cartesian"], find_q_labframe_cyl_to_cart(np.asarray(first)), atol=1e-12)
    assert not np.allclose(so["q_initial_cylindrical"], so["q_launch_cylindrical"])
