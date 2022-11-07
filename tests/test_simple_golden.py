from scotty.beam_me_up import beam_me_up
from scotty.init_bruv import get_parameters_for_Scotty
from scotty.generate_input import write_torbeam_file

import numpy as np
from numpy.testing import assert_allclose


TAU_EXPECTED = np.array(
    [
        0.0,
        62.647098,
        125.294195,
        187.941293,
        250.58839,
        281.93231,
        313.235488,
        375.882585,
        438.529683,
        501.176781,
    ]
)

B_EXPECTED = np.array(
    [
        0.758938,
        0.794796,
        0.82659,
        0.85399,
        0.877001,
        0.886924,
        0.895815,
        0.910728,
        0.922128,
        0.930421,
    ]
)

PSI_START_EXPECTED = np.array(
    [
        [
            -2.47359641e03 + 1.19557798e01j,
            1.40800267e-02 + 1.44252978e-02j,
            3.40526964e02 - 1.13752564e02j,
        ],
        [
            1.40800267e-02 + 1.44252978e-02j,
            4.24589207e03 + 4.35001002e03j,
            -2.22523411e-03 - 2.27980139e-03j,
        ],
        [
            3.40526964e02 - 1.13752564e02j,
            -2.22523411e-03 - 2.27980139e-03j,
            4.25775858e02 + 1.08229209e03j,
        ],
    ]
)

PSI_FINAL_EXPECTED = np.array(
    [
        [
            -1610.67883254 + 1.27655263e04j,
            -49.98203756 + 3.42882509e01j,
            -553.16605173 - 1.41526821e03j,
        ],
        [
            -49.98203756 + 3.42882509e01j,
            3549.99540711 + 1.91633120e03j,
            32.60030345 + 1.07767585e01j,
        ],
        [
            -553.16605173 - 1.41526821e03j,
            32.60030345 + 1.07767585e01j,
            2466.57382678 + 1.57149892e02j,
        ],
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
    assert_allclose(output["Psi_3D_output"][0, ...], PSI_START_EXPECTED, rtol=1e-2)
    assert_allclose(output["Psi_3D_output"][-1, ...], PSI_FINAL_EXPECTED, rtol=1e-2)


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
    assert_allclose(output["Psi_3D_output"][0, ...], PSI_START_EXPECTED, rtol=1e-2)
    # Slightly larger tolerance here, likely due to floating-point
    # precision of file-based input
    assert_allclose(output["Psi_3D_output"][-1, ...], PSI_FINAL_EXPECTED, rtol=1.5e-2)


def test_simple_check_torbeam_field(tmp_path):
    kwargs_dict = get_parameters_for_Scotty("DBS_synthetic")

    write_torbeam_file(
        major_radius=kwargs_dict["R_axis"],
        minor_radius=kwargs_dict["minor_radius_a"],
        B_toroidal_max=kwargs_dict["B_T_axis"],
        B_poloidal_max=kwargs_dict["B_p_a"],
        buffer_factor=1.1,
        x_grid_length=100,
        z_grid_length=100,
        torbeam_directory_path=tmp_path,
    )

    kwargs_dict["B_p_a"] = 0.10
    kwargs_dict["output_filename_suffix"] = "_Bpa0.10"
    kwargs_dict["output_path"] = str(tmp_path) + "/"
    kwargs_dict["figure_flag"] = False
    kwargs_dict["len_tau"] = 10
    kwargs_dict["find_B_method"] = "torbeam"
    kwargs_dict["magnetic_data_path"] = tmp_path

    beam_me_up(**kwargs_dict)

    assert len(list(tmp_path.glob("*.npz"))) == 4

    with np.load(tmp_path / "data_output_Bpa0.10.npz") as f:
        output = dict(f)

    assert_allclose(output["tau_array"], TAU_EXPECTED, rtol=1e-5)
    assert_allclose(output["B_magnitude"], B_EXPECTED, rtol=1e-5)
    assert_allclose(output["Psi_3D_output"][0, ...], PSI_START_EXPECTED, rtol=1e-2)
    assert_allclose(output["Psi_3D_output"][-1, ...], PSI_FINAL_EXPECTED, rtol=1e-2)
