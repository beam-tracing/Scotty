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
            -4.95188857e03 + 1.19557216e01j,
            2.36418979e-02 + 2.42219712e-02j,
            7.32184479e02 - 1.13752621e02j,
        ],
        [
            2.36418979e-02 + 2.42219712e-02j,
            4.24589069e03 + 4.35006709e03j,
            -3.73632150e-03 - 3.82799520e-03j,
        ],
        [
            7.32184479e02 - 1.13752621e02j,
            -3.73632150e-03 - 3.82799520e-03j,
            3.63874560e02 + 1.08229844e03j,
        ],
    ]
)
PSI_CUTOFF_EXPECTED = np.array(
    [
        [
            -6215.81890979 + 1112.79901382j,
            204.93411206 + 98.10435562j,
            -255.46251452 - 637.72534654j,
        ],
        [
            204.93411206 + 98.10435562j,
            4120.50413434 + 3227.10908421j,
            62.07312874 + 84.60016062j,
        ],
        [
            -255.46251452 - 637.72534654j,
            62.07312874 + 84.60016062j,
            2361.82932675 + 370.99445929j,
        ],
    ]
)
PSI_FINAL_EXPECTED = np.array(
    [
        [
            9354.39965447 + 2934.80061896j,
            -44.43214182 - 41.86849138j,
            -123.74294699 + 935.41986947j,
        ],
        [
            -44.43214182 - 41.86849138j,
            3955.35931399 + 2700.40411176j,
            34.51990504 + 19.60417504j,
        ],
        [
            -123.74294699 + 935.41986947j,
            34.51990504 + 19.60417504j,
            3321.5750101 + 295.3422719j,
        ],
    ]
)

EXPECTED = {
    "q_R_array": np.array(
        [
            1.99383268,
            1.91861483,
            1.86523022,
            1.82681413,
            1.79978598,
            1.7900192,
            1.78240033,
            1.77411764,
            1.77542478,
            1.78803898,
        ]
    ),
    "q_Z_array": np.array(
        [
            -0.0780444,
            -0.0899398,
            -0.10643088,
            -0.12729439,
            -0.15263795,
            -0.16698436,
            -0.18291117,
            -0.21905541,
            -0.26281654,
            -0.3173612,
        ]
    ),
    "q_zeta_array": np.array(
        [
            0.0,
            0.00042848,
            0.00137181,
            0.00255236,
            0.00383913,
            0.00449193,
            0.00515602,
            0.00644165,
            0.00762059,
            0.00856271,
        ]
    ),
    "K_R_array": np.array(
        [
            -1146.40007,
            -800.81279303,
            -575.00626934,
            -411.65453026,
            -280.8558433,
            -222.17217241,
            -164.23423963,
            -47.97530652,
            81.86606319,
            244.41372078,
        ]
    ),
    "K_Z_array": np.array(
        [
            -120.49150267,
            -183.80742948,
            -240.31574029,
            -295.47106115,
            -353.89542525,
            -385.69536967,
            -421.08544016,
            -505.02703605,
            -618.86935667,
            -786.33164194,
        ]
    ),
    "K_zeta_initial": np.array(-0.0),
    "tau_array": np.array(
        [
            0.0,
            52.04326145,
            104.0865229,
            156.12978435,
            208.1730458,
            233.97324234,
            260.21630725,
            312.2595687,
            364.30283015,
            416.3460916,
        ]
    ),
    "B_magnitude": np.array(
        [
            0.7589359,
            0.78648985,
            0.80778142,
            0.8240924,
            0.83614377,
            0.84064856,
            0.84424788,
            0.84839825,
            0.84829216,
            0.84327565,
        ]
    ),
    "normalised_gyro_freqs": np.array(
        [
            0.38626373,
            0.40028744,
            0.41112388,
            0.41942542,
            0.42555902,
            0.42785175,
            0.42968364,
            0.43179599,
            0.431742,
            0.42918882,
        ]
    ),
    "normalised_plasma_freqs": np.array(
        [
            0.01808471,
            0.70197289,
            0.84189027,
            0.89884012,
            0.92062895,
            0.9230766,
            0.92055248,
            0.89857289,
            0.84127185,
            0.70037915,
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
    assert_allclose(output["Psi_3D_output"][-1, ...], PSI_FINAL_EXPECTED, rtol=2e-2)
