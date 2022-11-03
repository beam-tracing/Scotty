from scotty.beam_me_up import beam_me_up
from scotty.init_bruv import get_parameters_for_Scotty

import numpy as np
from numpy.testing import assert_allclose


TAU_EXPECTED = np.array(
    [
        0.0,
        62.69833027,
        125.39666054,
        188.09499082,
        250.79332109,
        282.12565681,
        313.49165136,
        376.18998163,
        438.8883119,
        501.58664217,
    ]
)
B_EXPECTED = np.array(
    [
        0.75893835,
        0.79481524,
        0.82660747,
        0.85400248,
        0.87701708,
        0.88693792,
        0.89586574,
        0.91087338,
        0.92241292,
        0.93088217,
    ]
)


def test_simple_check(tmp_path):
    """This does a really simple golden answer test to check basic functionality"""

    kwargs_dict = get_parameters_for_Scotty("DBS_synthetic")
    kwargs_dict["B_p_a"] = 0.10
    kwargs_dict["output_filename_suffix"] = "_Bpa0.10"
    kwargs_dict["output_path"] = str(tmp_path) + "/"
    kwargs_dict["figure_flag"] = False
    kwargs_dict["len_tau"] = 10

    beam_me_up(**kwargs_dict)

    assert len(list(tmp_path.glob("*.npz"))) == 4

    with np.load(tmp_path / "data_output_Bpa0.10.npz") as f:
        output = dict(f)

    assert_allclose(output["tau_array"], TAU_EXPECTED, rtol=1e-5)
    assert_allclose(output["B_magnitude"], B_EXPECTED, rtol=1e-5)


def test_simple_check_density_fit_file(tmp_path):
    """Similar to the simple check, but the density fit is from file instead"""

    kwargs_dict = get_parameters_for_Scotty("DBS_synthetic")

    ne_filename = tmp_path / "ne.dat"
    rho = np.linspace(0, 1)
    density_fit = kwargs_dict["density_fit_method"]
    density = density_fit(rho**2)

    with open(ne_filename, "w") as f:
        f.write(f"{len(rho)}\n")
        np.savetxt(f, np.column_stack((rho, density)), fmt="%.7e")

    kwargs_dict["B_p_a"] = 0.10
    kwargs_dict["output_filename_suffix"] = "_Bpa0.10"
    kwargs_dict["output_path"] = str(tmp_path) + "/"
    kwargs_dict["figure_flag"] = False
    kwargs_dict["len_tau"] = 10
    kwargs_dict["density_fit_parameters"] = None
    kwargs_dict["density_fit_method"] = "smoothing-spline-file"
    kwargs_dict["ne_data_path"] = tmp_path

    beam_me_up(**kwargs_dict)

    assert len(list(tmp_path.glob("*.npz"))) == 4

    with np.load(tmp_path / "data_output_Bpa0.10.npz") as f:
        output = dict(f)

    assert_allclose(output["tau_array"], TAU_EXPECTED, rtol=1e-5)
    assert_allclose(output["B_magnitude"], B_EXPECTED, rtol=1e-5)
