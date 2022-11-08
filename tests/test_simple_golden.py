from scotty.beam_me_up import beam_me_up
from scotty.init_bruv import get_parameters_for_Scotty
from scotty.torbeam import write_torbeam_file

import numpy as np
from numpy.testing import assert_allclose
import pytest


CUTOFF_INDEX = 5

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
PSI_CUTOFF_EXPECTED = np.array(
    [
        [
            -3415.6309789 + 435.2610082j,
            93.82701509 + 44.60665203j,
            491.12733009 - 402.19521605j,
        ],
        [
            93.82701509 + 44.60665203j,
            3949.71530077 + 2687.84876321j,
            58.35671659 + 57.58346865j,
        ],
        [
            491.12733009 - 402.19521605j,
            58.35671659 + 57.58346865j,
            2064.03965809 + 377.61804618j,
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

EXPECTED = {
    "q_R_array": np.array(
        [
            1.99382564,
            1.89728654,
            1.82082333,
            1.76059051,
            1.71360145,
            1.69429565,
            1.67748372,
            1.65044156,
            1.63108651,
            1.61840018,
        ]
    ),
    "q_Z_array": np.array(
        [
            -0.07804514,
            -0.09140912,
            -0.10936584,
            -0.13290859,
            -0.16326418,
            -0.18147609,
            -0.20197093,
            -0.25095695,
            -0.31266471,
            -0.39018781,
        ]
    ),
    "q_zeta_array": np.array(
        [
            0.0,
            0.00035299,
            0.00123228,
            0.00243089,
            0.00380219,
            0.00451398,
            0.00522458,
            0.00659428,
            0.00779435,
            0.00867671,
        ]
    ),
    "K_R_array": np.array(
        [
            -1146.40007,
            -909.51030079,
            -718.39083261,
            -563.41926333,
            -436.88422749,
            -382.19530218,
            -332.38822093,
            -244.73125986,
            -169.47574434,
            -102.83201513,
        ]
    ),
    "K_Z_array": np.array(
        [
            -120.49150267,
            -165.70906882,
            -219.43741364,
            -284.25967555,
            -363.43816465,
            -409.66390866,
            -460.98220458,
            -581.7805972,
            -731.89716718,
            -918.89504231,
        ]
    ),
    "K_zeta_initial": np.array(-0.0),
    "tau_array": np.array(
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
    ),
    "B_magnitude": np.array(
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
    ),
    "normalised_gyro_freqs": np.array(
        [
            0.38626498,
            0.40451481,
            0.4206954,
            0.4346436,
            0.44635263,
            0.4514013,
            0.45592739,
            0.46351986,
            0.46932146,
            0.47354238,
        ]
    ),
    "normalised_plasma_freqs": np.array(
        [
            0.01388158,
            0.59779342,
            0.75902333,
            0.83733011,
            0.87051483,
            0.87441588,
            0.87048555,
            0.8372284,
            0.75887052,
            0.59753335,
        ]
    ),
}


def simple(path):
    """Built-in synthetic diagnostic"""
    return get_parameters_for_Scotty("DBS_synthetic")


def ne_dat_file(path):
    """Density fit using TORBEAM file"""
    kwargs_dict = get_parameters_for_Scotty("DBS_synthetic")

    ne_filename = path / "ne.dat"
    rho = np.linspace(0, 1)
    density_fit = kwargs_dict["density_fit_method"]
    density = density_fit(rho**2)

    with open(ne_filename, "w") as f:
        f.write(f"{len(rho)}\n")
        np.savetxt(f, np.column_stack((rho, density)), fmt="%.7e")

    kwargs_dict["density_fit_parameters"] = None
    kwargs_dict["density_fit_method"] = "smoothing-spline-file"
    kwargs_dict["ne_data_path"] = path
    return kwargs_dict


def torbeam_file(path):
    """Geometry using TORBEAM file"""
    kwargs_dict = get_parameters_for_Scotty("DBS_synthetic")

    write_torbeam_file(
        major_radius=kwargs_dict["R_axis"],
        minor_radius=kwargs_dict["minor_radius_a"],
        B_toroidal_max=kwargs_dict["B_T_axis"],
        B_poloidal_max=kwargs_dict["B_p_a"],
        buffer_factor=1.1,
        x_grid_length=100,
        z_grid_length=100,
        torbeam_directory_path=path,
    )

    kwargs_dict["find_B_method"] = "torbeam"
    kwargs_dict["magnetic_data_path"] = path
    return kwargs_dict


@pytest.mark.parametrize(
    "generator",
    [
        pytest.param(simple, id="simple"),
        pytest.param(ne_dat_file, id="density-fit-file"),
        pytest.param(torbeam_file, id="torbeam-file"),
    ],
)
def test_integrated(tmp_path, generator):
    """Golden answer test to check basic functionality using circular
    flux surfaces."""

    kwargs_dict = generator(tmp_path)
    kwargs_dict["B_p_a"] = 0.10
    kwargs_dict["output_filename_suffix"] = "_Bpa0.10"
    kwargs_dict["figure_flag"] = False
    kwargs_dict["len_tau"] = 10
    kwargs_dict["output_path"] = tmp_path

    beam_me_up(**kwargs_dict)

    assert len(list(tmp_path.glob("*.npz"))) == 4

    with np.load(tmp_path / "data_output_Bpa0.10.npz") as f:
        output = dict(f)

    for key, value in EXPECTED.items():
        assert_allclose(output[key], value, rtol=1e-2, err_msg=key)

    K_magnitude = np.hypot(output["K_R_array"], output["K_Z_array"])
    assert K_magnitude.argmin() == CUTOFF_INDEX

    assert_allclose(output["Psi_3D_output"][0, ...], PSI_START_EXPECTED, rtol=1e-2)
    assert_allclose(
        output["Psi_3D_output"][CUTOFF_INDEX, ...], PSI_CUTOFF_EXPECTED, rtol=1e-2
    )
    # Slightly larger tolerance here, likely due to floating-point
    # precision of file-based input
    assert_allclose(output["Psi_3D_output"][-1, ...], PSI_FINAL_EXPECTED, rtol=1.5e-2)
