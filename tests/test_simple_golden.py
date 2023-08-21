from scotty.beam_me_up import (
    beam_me_up,
    create_magnetic_geometry,
    make_density_fit,
)
from scotty.init_bruv import get_parameters_for_Scotty
from scotty.torbeam import Torbeam
from scotty.geometry import CircularCrossSectionField
from scotty.fun_general import (
    freq_GHz_to_angular_frequency,
    angular_frequency_to_wavenumber,
)
from scotty.profile_fit import QuadraticFit, TanhFit, PolynomialFit, ProfileFit
from scotty.hamiltonian import Hamiltonian
from scotty.launch import find_entry_point, launch_beam
from scotty.typing import FloatArray

import json
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pathlib
import pytest

# Print more of arrays in failed tests
np.set_printoptions(linewidth=120, threshold=100)

# For circular flux surfaces, O-mode mode_flag=1, X-mode mode_flag=-1

# Expected values for nonrelativistic simple golden, mode_flag = 1

CUTOFF_INDEX_1 = 5

PSI_START_EXPECTED_1 = np.array(
    [
        [
            -2.47357794e03 + 1.19557806e01j,
            -1.39916828e-02 - 1.43347874e-02j,
            3.40523926e02 - 1.13752568e02j,
        ],
        [
            -1.39916828e-02 - 1.43347874e-02j,
            4.24589207e03 + 4.35001002e03j,
            2.21127206e-03 + 2.26549697e-03j,
        ],
        [
            3.40523926e02 - 1.13752568e02j,
            2.21127206e-03 + 2.26549697e-03j,
            4.25776357e02 + 1.08229209e03j,
        ],
    ]
)
PSI_CUTOFF_EXPECTED_1 = np.array(
    [
        [
            -3415.07623355 + 445.31992622j,
            -282.82924665 - 149.90119128j,
            509.18231115 - 398.50960651j,
        ],
        [
            -282.82924665 - 149.90119128j,
            3929.38936698 + 2642.61848492j,
            -55.70975677 - 74.68041153j,
        ],
        [
            509.18231115 - 398.50960651j,
            -55.70975677 - 74.68041153j,
            2049.26382135 + 377.83003374j,
        ],
    ]
)
PSI_FINAL_EXPECTED_1 = np.array(
    [
        [
            -759.55423192 + 13243.111223j,
            -479.31204769 + 514.09773505j,
            -644.55622419 - 1452.45488266j,
        ],
        [
            -479.31204769 + 514.09773505j,
            3487.94478663 + 1885.63739326j,
            12.71272627 - 77.55460115j,
        ],
        [
            -644.55622419 - 1452.45488266j,
            12.71272627 - 77.55460115j,
            2470.84658744 + 159.68128543j,
        ],
    ]
)


EXPECTED_1 = {
    "q_R_array": np.array(
        [
            1.99382564,
            1.8973136,
            1.82080896,
            1.76044632,
            1.71327164,
            1.69395316,
            1.67708778,
            1.65019221,
            1.63108066,
            1.61859347,
        ]
    ),
    "q_Z_array": np.array(
        [
            -0.07804514,
            -0.09137557,
            -0.10919534,
            -0.13253917,
            -0.1627768,
            -0.18095369,
            -0.20155828,
            -0.25072523,
            -0.3125676,
            -0.3901351,
        ]
    ),
    "q_zeta_array": np.array(
        [
            0.0,
            -0.00041565,
            -0.00168641,
            -0.00380195,
            -0.00660756,
            -0.00815794,
            -0.00972988,
            -0.01265907,
            -0.01496556,
            -0.01641859,
        ]
    ),
    "K_R_array": np.array(
        [
            -1146.40007,
            -909.7351259,
            -718.97020274,
            -564.32641187,
            -437.85637809,
            -383.2161993,
            -333.12907589,
            -245.07498321,
            -169.39921007,
            -102.43869197,
        ]
    ),
    "K_Z_array": np.array(
        [
            -120.49150267,
            -165.72589293,
            -219.64365799,
            -284.80168007,
            -364.33147671,
            -410.55892646,
            -462.07152722,
            -582.78670158,
            -732.58798218,
            -919.20759736,
        ]
    ),
    "K_zeta_initial": np.array(-0.0),
    "tau_array": np.array(
        [
            0.0,
            62.6209717,
            125.2419434,
            187.86291511,
            250.48388681,
            281.73236206,
            313.10485851,
            375.72583021,
            438.34680191,
            500.96777362,
        ]
    ),
    "B_magnitude": np.array(
        [
            0.75898041,
            0.79758802,
            0.83110015,
            0.85959713,
            0.883266,
            0.8933391,
            0.90232283,
            0.91702929,
            0.92777423,
            0.93493185,
        ]
    ),
    "normalised_gyro_freqs": np.array(
        [
            0.38628639,
            0.4059359,
            0.42299204,
            0.43749571,
            0.44954208,
            0.45466883,
            0.45924114,
            0.46672606,
            0.47219474,
            0.47583764,
        ]
    ),
    "normalised_plasma_freqs": np.array(
        [
            0.0,
            0.59773859,
            0.75915389,
            0.83777097,
            0.8712485,
            0.87520137,
            0.8712369,
            0.83771518,
            0.7590454,
            0.59751663,
        ]
    ),
}

# Expected values for nonrelativistic simple golden, mode_flag = -1

CUTOFF_INDEX_NEG1 = 5

PSI_START_EXPECTED_NEG1 = np.array(
    [
        [
            -2.91536252e03 + 1.19564545e01j,
            -1.70148132e-15 - 2.21090894e-15j,
            4.10273717e02 - 1.13758066e02j,
        ],
        [
            -1.70148132e-15 - 2.21090894e-15j,
            4.24588292e03 + 4.35038716e03j,
            1.61885134e-14 + 2.10353935e-14j,
        ],
        [
            4.10273717e02 - 1.13758066e02j,
            1.61885134e-14 + 2.10353935e-14j,
            4.14727090e02 + 1.08233570e03j,
        ],
    ]
)
PSI_CUTOFF_EXPECTED_NEG1 = np.array(
    [
        [
            -7.73550781e04 + 1.34722684e04j,
            9.39057362e02 + 2.50803785e02j,
            1.46623251e04 - 5.02535302e03j,
        ],
        [
            9.39057362e02 + 2.50803785e02j,
            3.56049986e03 + 1.93889334e03j,
            -1.18904163e02 + 3.27447497e01j,
        ],
        [
            1.46623251e04 - 5.02535302e03j,
            -1.18904163e02 + 3.27447497e01j,
            -1.47350536e02 + 1.88270173e03j,
        ],
    ]
)
PSI_FINAL_EXPECTED_NEG1 = np.array(
    [
        [
            5.02917079e03 + 1.35480590e02j,
            2.34748730e01 + 1.58346740e01j,
            1.93577117e02 + 1.11821703e02j,
        ],
        [
            2.34748730e01 + 1.58346740e01j,
            2.87217332e03 + 1.11883142e03j,
            4.20742523e01 + 2.19936124e01j,
        ],
        [
            1.93577117e02 + 1.11821703e02j,
            4.20742523e01 + 2.19936124e01j,
            2.59750859e03 + 9.26939766e01j,
        ],
    ]
)


EXPECTED_NEG1 = {
    "q_R_array": np.array(
        [
            1.99387217,
            1.93805575,
            1.89170709,
            1.85602899,
            1.83317026,
            1.82688086,
            1.82498201,
            1.83126083,
            1.84988849,
            1.87844182,
        ]
    ),
    "q_Z_array": np.array(
        [
            -0.07804025,
            -0.08519773,
            -0.09424295,
            -0.10582153,
            -0.12098197,
            -0.13112771,
            -0.14102312,
            -0.16698944,
            -0.19882757,
            -0.23601108,
        ]
    ),
    "q_zeta_array": np.array(
        [
            3.64302241e-17,
            1.56090264e-04,
            6.50453332e-04,
            1.51093011e-03,
            2.69111683e-03,
            3.39990235e-03,
            3.99935338e-03,
            5.16680571e-03,
            6.00991893e-03,
            6.49247319e-03,
        ]
    ),
    "K_R_array": np.array(
        [
            -1146.40007,
            -971.90930973,
            -784.6638796,
            -564.7722809,
            -299.43104475,
            -143.44677255,
            -11.7005995,
            248.24017306,
            456.45793455,
            626.21158363,
        ]
    ),
    "K_Z_array": np.array(
        [
            -120.49150267,
            -150.83839501,
            -190.26794372,
            -246.4010787,
            -328.33628241,
            -383.74064007,
            -434.87555229,
            -548.7379687,
            -654.86418912,
            -753.33397823,
        ]
    ),
    "K_zeta_initial": np.array(-0.0),
    "tau_array": np.array(
        [
            0.0,
            35.00328799,
            70.00657599,
            105.00986398,
            140.01315198,
            158.93919148,
            175.01643997,
            210.01972796,
            245.02301596,
            280.02630395,
        ]
    ),
    "B_magnitude": np.array(
        [
            0.75896269,
            0.78082098,
            0.79995185,
            0.81532918,
            0.82549594,
            0.82833787,
            0.82919973,
            0.82635667,
            0.81803557,
            0.80560099,
        ]
    ),
    "normalised_gyro_freqs": np.array(
        [
            0.38627737,
            0.39740224,
            0.40713899,
            0.41496534,
            0.42013976,
            0.42158617,
            0.42202482,
            0.42057783,
            0.41634277,
            0.41001414,
        ]
    ),
    "normalised_plasma_freqs": np.array(
        [
            -0.0,
            0.46563763,
            0.61146091,
            0.69120285,
            0.72820049,
            0.73285099,
            0.72860508,
            0.69223732,
            0.61272118,
            0.4666989,
        ]
    ),
}

# Expected values for relativistic version of the test
# Note: expected values are based on a tentative version
# of the relativistic code and are not yet rigorously
# cross-validated. May be subject to changes in the future.

# For mode_flag = 1:

CUTOFF_INDEX_REL_1 = 5

PSI_START_EXPECTED_REL_1 = np.array(
    [
        [
            -2.47380536e03 + 1.19564545e01j,
            -1.70148132e-15 - 2.21090894e-15j,
            3.40499803e02 - 1.13758066e02j,
        ],
        [
            -1.70148132e-15 - 2.21090894e-15j,
            4.24588292e03 + 4.35038716e03j,
            1.61885134e-14 + 2.10353935e-14j,
        ],
        [
            3.40499803e02 - 1.13758066e02j,
            1.61885134e-14 + 2.10353935e-14j,
            4.25752614e02 + 1.08233570e03j,
        ],
    ]
)
PSI_CUTOFF_EXPECTED_REL_1 = np.array(  # Not yet modified
    [
        [
            -2.74095960e03 + 3.58127405e02j,
            -2.65492476e02 - 1.41323059e02j,
            3.88466862e02 - 3.48400619e02j,
        ],
        [
            -2.65492476e02 - 1.41323059e02j,
            3.90471855e03 + 2.58339866e03j,
            -5.91261562e01 - 7.33060225e01j,
        ],
        [
            3.88466862e02 - 3.48400619e02j,
            -5.91261562e01 - 7.33060225e01j,
            2.02222919e03 + 3.57401128e02j,
        ],
    ]
)
PSI_FINAL_EXPECTED_REL_1 = np.array(
    [
        [
            -3.71494845e03 + 3.81425685e03j,
            -4.42926587e02 + 2.66468589e01j,
            1.57646260e02 - 7.14883020e02j,
        ],
        [
            -4.42926587e02 + 2.66468589e01j,
            3.44023583e03 + 1.78008390e03j,
            4.19741443e01 - 2.61204714e01j,
        ],
        [
            1.57646260e02 - 7.14883020e02j,
            4.19741443e01 - 2.61204714e01j,
            2.35535115e03 + 1.34097969e02j,
        ],
    ]
)


EXPECTED_REL_1 = {
    "q_R_array": np.array(
        [
            1.99387217,
            1.89478605,
            1.81608475,
            1.75333963,
            1.70332833,
            1.68236306,
            1.66365248,
            1.63245724,
            1.6081207,
            1.58926036,
        ]
    ),
    "q_Z_array": np.array(
        [
            -0.07804025,
            -0.09178393,
            -0.11014496,
            -0.13409806,
            -0.16497459,
            -0.18346808,
            -0.20440327,
            -0.25422772,
            -0.31682193,
            -0.39541607,
        ]
    ),
    "q_zeta_array": np.array(
        [
            3.64302241e-17,
            -4.34557216e-04,
            -1.75458750e-03,
            -3.94757309e-03,
            -6.87025031e-03,
            -8.48770015e-03,
            -1.01300439e-02,
            -1.31890816e-02,
            -1.56082400e-02,
            -1.71395528e-02,
        ]
    ),
    "K_R_array": np.array(
        [
            -1146.40007,
            -907.90136177,
            -722.04856355,
            -575.25018894,
            -458.16628298,
            -408.60089775,
            -363.80900807,
            -287.14391725,
            -224.28323101,
            -172.04458472,
        ]
    ),
    "K_Z_array": np.array(
        [
            -120.49150267,
            -166.27151556,
            -219.69675587,
            -283.52247217,
            -360.9422406,
            -405.84935937,
            -455.91708955,
            -573.53506261,
            -720.72904123,
            -906.95011315,
        ]
    ),
    "K_zeta_initial": np.array(-0.0),
    "tau_array": np.array(
        [
            0.0,
            64.39979252,
            128.79958503,
            193.19937755,
            257.59917007,
            289.73242526,
            321.99896258,
            386.3987551,
            450.79854762,
            515.19834013,
        ]
    ),
    "B_magnitude": np.array(
        [
            0.75896269,
            0.79865196,
            0.8332621,
            0.86308127,
            0.88842214,
            0.89949347,
            0.9096098,
            0.92699187,
            0.94102053,
            0.95218797,
        ]
    ),
    "normalised_gyro_freqs": np.array(
        [
            0.38627737,
            0.39977178,
            0.41294464,
            0.42530111,
            0.43666288,
            0.44196733,
            0.44707702,
            0.45679554,
            0.46634932,
            0.47662924,
        ]
    ),
    "normalised_plasma_freqs": np.array(
        [
            -0.0,
            0.59956615,
            0.756831,
            0.83240653,
            0.86437198,
            0.86814015,
            0.86436323,
            0.83237437,
            0.75676491,
            0.59942523,
        ]
    ),
}

# For mode_flag = -1:

CUTOFF_INDEX_REL_NEG1 = 5

PSI_START_EXPECTED_REL_NEG1 = np.array(
    [
        [
            -2.91536253e03 + 1.19564545e01j,
            -1.70148132e-15 - 2.21090894e-15j,
            4.10273883e02 - 1.13758066e02j,
        ],
        [
            -1.70148132e-15 - 2.21090894e-15j,
            4.24588292e03 + 4.35038716e03j,
            1.61885134e-14 + 2.10353935e-14j,
        ],
        [
            4.10273883e02 - 1.13758066e02j,
            1.61885134e-14 + 2.10353935e-14j,
            4.14727038e02 + 1.08233570e03j,
        ],
    ]
)
PSI_CUTOFF_EXPECTED_REL_NEG1 = np.array(  # Not yet modified
    [
        [
            -6.16175436e04 + 1.03713923e04j,
            8.12739539e02 + 2.10348335e02j,
            1.16247793e04 - 4.06046898e03j,
        ],
        [
            8.12739539e02 + 2.10348335e02j,
            3.49517564e03 + 1.84058019e03j,
            -8.99870324e01 + 3.81172348e01j,
        ],
        [
            1.16247793e04 - 4.06046898e03j,
            -8.99870324e01 + 3.81172348e01j,
            3.94919303e02 + 1.59793834e03j,
        ],
    ]
)
PSI_FINAL_EXPECTED_REL_NEG1 = np.array(
    [
        [
            5.00629634e03 + 1.41790111e02j,
            2.37427524e01 + 1.53467963e01j,
            1.32572176e02 + 1.06636090e02j,
        ],
        [
            2.37427524e01 + 1.53467963e01j,
            2.77833196e03 + 1.03516633e03j,
            4.05478764e01 + 2.00937539e01j,
        ],
        [
            1.32572176e02 + 1.06636090e02j,
            4.05478764e01 + 2.00937539e01j,
            2.54989309e03 + 8.20268367e01j,
        ],
    ]
)


EXPECTED_REL_NEG1 = {
    "q_R_array": np.array(
        [
            1.99387217,
            1.93488909,
            1.88624692,
            1.84893205,
            1.8248492,
            1.81789931,
            1.81557836,
            1.82087145,
            1.83880461,
            1.86713704,
        ]
    ),
    "q_Z_array": np.array(
        [
            -0.07804025,
            -0.08568082,
            -0.09540517,
            -0.10787861,
            -0.12417613,
            -0.13522058,
            -0.14563565,
            -0.17340041,
            -0.20758102,
            -0.24778288,
        ]
    ),
    "q_zeta_array": np.array(
        [
            3.64302241e-17,
            1.74299842e-04,
            7.21552511e-04,
            1.66291745e-03,
            2.94204754e-03,
            3.71818533e-03,
            4.35401195e-03,
            5.61994726e-03,
            6.54366208e-03,
            7.07804388e-03,
        ]
    ),
    "K_R_array": np.array(
        [
            -1146.40007,
            -964.15226622,
            -774.08556804,
            -557.08686451,
            -302.27204486,
            -152.32308785,
            -29.81061679,
            218.50784263,
            421.85750003,
            591.35208292,
        ]
    ),
    "K_Z_array": np.array(
        [
            -120.49150267,
            -152.3625245,
            -193.08753826,
            -250.12943991,
            -332.07564378,
            -388.04439146,
            -438.25600862,
            -553.79684312,
            -664.74415576,
            -770.67097607,
        ]
    ),
    "K_zeta_initial": np.array(-0.0),
    "tau_array": np.array(
        [
            0.0,
            37.14107905,
            74.2821581,
            111.42323715,
            148.5643162,
            168.93919,
            185.70539526,
            222.84647431,
            259.98755336,
            297.12863241,
        ]
    ),
    "B_magnitude": np.array(
        [
            0.75896269,
            0.78209888,
            0.8022675,
            0.81845874,
            0.82926008,
            0.83243037,
            0.83349451,
            0.83107162,
            0.8229665,
            0.81047859,
        ]
    ),
    "normalised_gyro_freqs": np.array(
        [
            0.38627737,
            0.39392634,
            0.40111904,
            0.40726518,
            0.41165983,
            0.4131073,
            0.41375025,
            0.41351246,
            0.41143571,
            0.40819815,
        ]
    ),
    "normalised_plasma_freqs": np.array(
        [
            -0.0,
            0.47527609,
            0.61980874,
            0.69722851,
            0.73261861,
            0.73703945,
            0.7330262,
            0.698295,
            0.62116946,
            0.47652103,
        ]
    ),
}


def simple(path):
    """Built-in synthetic diagnostic"""
    kwargs_dict = get_parameters_for_Scotty("DBS_synthetic")
    kwargs_dict["find_B_method"] = "unit-tests"
    return kwargs_dict


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

    kwargs_dict["find_B_method"] = "unit-tests"
    kwargs_dict["density_fit_parameters"] = None
    kwargs_dict["density_fit_method"] = "smoothing-spline-file"
    kwargs_dict["ne_data_path"] = path
    return kwargs_dict


def synthetic_args():
    kwargs_dict = get_parameters_for_Scotty("DBS_synthetic")
    field = CircularCrossSectionField(
        R_axis=kwargs_dict["R_axis"],
        minor_radius_a=kwargs_dict["minor_radius_a"],
        B_T_axis=kwargs_dict["B_T_axis"],
        B_p_a=kwargs_dict["B_p_a"],
        R_points=256,
        Z_points=256,
        grid_buffer_factor=1.1,
    )
    x_meshgrid, z_meshgrid = np.meshgrid(field.R_coord, field.Z_coord, indexing="ij")
    B_t = field.B_T(x_meshgrid, z_meshgrid)
    B_r = field.B_R(x_meshgrid, z_meshgrid)
    B_z = field.B_Z(x_meshgrid, z_meshgrid)
    psi = field.poloidal_flux(x_meshgrid, z_meshgrid)
    return kwargs_dict, B_r, B_t, B_z, psi, field


def torbeam_file(path):
    """Geometry using TORBEAM file"""
    kwargs_dict, B_R, B_t, B_Z, psi, field = synthetic_args()
    x_meshgrid, z_meshgrid = np.meshgrid(field.R_coord, field.Z_coord, indexing="ij")
    B_t = field.B_T(x_meshgrid, z_meshgrid)
    B_r = field.B_R(x_meshgrid, z_meshgrid)
    B_z = field.B_Z(x_meshgrid, z_meshgrid)
    psi = field.poloidal_flux(x_meshgrid, z_meshgrid)

    Torbeam(field.R_coord, field.Z_coord, B_r, B_t, B_z, psi).write(path / "topfile")

    kwargs_dict["find_B_method"] = "torbeam"
    kwargs_dict["magnetic_data_path"] = path
    return kwargs_dict


def omfit_json(path):
    """Geometry using OMFIT JSON file"""
    kwargs_dict, B_R, B_t, B_Z, psi, field = synthetic_args()

    def flatten_as_F(array):
        return array.T.flatten().tolist()

    data = {
        "R": field.R_coord.tolist(),
        "Z": field.Z_coord.tolist(),
        "Br": flatten_as_F(B_R),
        "Bt": flatten_as_F(B_t),
        "Bz": flatten_as_F(B_Z),
        "pol_flux": flatten_as_F(psi),
    }

    with open(path / "topfile.json", "w") as f:
        json.dump(data, f)

    kwargs_dict["find_B_method"] = "omfit"
    kwargs_dict["magnetic_data_path"] = path
    return kwargs_dict


def add_timeslices(array):
    """Add a new dimension at the front, with 'array' at index 1"""
    zeros = np.zeros_like(array)
    return np.stack((zeros, array, zeros))


def npz_file(path: pathlib.Path):
    """find_B_method = "test" """

    kwargs_dict, B_R, B_t, B_Z, psi, field = synthetic_args()

    time = [100, 200, 300]

    shot = 12345
    np.savez(
        path / f"{shot}_equilibrium_data",
        R_EFIT=field.R_coord,
        Z_EFIT=field.Z_coord,
        poloidalFlux_grid=add_timeslices(psi),
        Bphi_grid=add_timeslices(B_t),
        Br_grid=add_timeslices(B_R),
        Bz_grid=add_timeslices(B_Z),
        time_EFIT=time,
    )
    kwargs_dict["find_B_method"] = "test"
    kwargs_dict["magnetic_data_path"] = path
    kwargs_dict["shot"] = shot
    kwargs_dict["equil_time"] = time[1]

    return kwargs_dict


def npz_notime_file(path: pathlib.Path):
    """find_B_method = "test_notime" """
    kwargs_dict, B_R, B_t, B_Z, psi, field = synthetic_args()

    np.savez(
        path / "12345_equilibrium_data",
        R_EFIT=field.R_coord,
        Z_EFIT=field.Z_coord,
        poloidalFlux_grid=psi,
        Bphi_grid=B_t,
        Br_grid=B_R,
        Bz_grid=B_Z,
    )
    kwargs_dict["find_B_method"] = "test_notime"
    kwargs_dict["magnetic_data_path"] = path / "12345_equilibrium_data.npz"
    return kwargs_dict


def UDA_saved(path: pathlib.Path):
    """find_B_method = "UDA_saved" with shot = None"""
    kwargs_dict, B_R, B_t, B_Z, psi, field = synthetic_args()

    R_axis = kwargs_dict["R_axis"]
    minor_radius = kwargs_dict["minor_radius_a"]
    B_p_a = kwargs_dict["B_p_a"]
    psi_axis = field.poloidal_flux(R_axis, 0.0) * B_p_a
    psi_boundary = field.poloidal_flux(R_axis + minor_radius, 0.0) * B_p_a

    R_midplane = np.linspace(R_axis, R_axis + minor_radius)
    rBphi = R_midplane * field.B_T(R_midplane, 0.0)

    np.savez(
        path / "12345_equilibrium_data",
        rBphi=rBphi,
        R_EFIT=field.R_coord,
        Z_EFIT=field.Z_coord,
        poloidalFlux_grid=psi,
        poloidalFlux_unnormalised_axis=-psi_axis,
        poloidalFlux_unnormalised_boundary=-psi_boundary,
    )
    kwargs_dict["find_B_method"] = "UDA_saved"
    kwargs_dict["magnetic_data_path"] = path / "12345_equilibrium_data.npz"
    kwargs_dict["shot"] = None
    kwargs_dict["delta_R"] = -0.0001
    kwargs_dict["delta_Z"] = 0.0001
    return kwargs_dict


def UDA_saved_MAST_U(path: pathlib.Path):
    """find_B_method = "UDA_saved" with shot = None"""
    kwargs_dict, B_R, B_t, B_Z, psi, field = synthetic_args()

    shot = 55555

    R_axis = kwargs_dict["R_axis"]
    minor_radius = kwargs_dict["minor_radius_a"]
    B_p_a = kwargs_dict["B_p_a"]
    psi_axis = field.poloidal_flux(R_axis, 0.0) * B_p_a
    psi_boundary = field.poloidal_flux(R_axis + minor_radius, 0.0) * B_p_a

    R_midplane = np.linspace(R_axis, R_axis + minor_radius)
    rBphi = R_midplane * field.B_T(R_midplane, 0.0)
    psi_norm_1D = np.linspace(0, 1, len(R_midplane))

    time = [100, 200, 300]

    np.savez(
        path / f"{shot}_equilibrium_data",
        rBphi=add_timeslices(rBphi),
        R_EFIT=field.R_coord,
        Z_EFIT=field.Z_coord,
        poloidalFlux_grid=add_timeslices(psi),
        poloidalFlux=add_timeslices(psi_norm_1D),
        poloidalFlux_unnormalised_axis=add_timeslices(-psi_axis),
        poloidalFlux_unnormalised_boundary=add_timeslices(-psi_boundary),
        time_EFIT=time,
    )
    kwargs_dict["find_B_method"] = "UDA_saved"
    kwargs_dict["magnetic_data_path"] = path
    kwargs_dict["shot"] = shot
    kwargs_dict["delta_R"] = -0.0001
    kwargs_dict["delta_Z"] = 0.0001
    kwargs_dict["equil_time"] = time[1]
    return kwargs_dict


def assert_is_symmetric(array: FloatArray) -> None:
    assert_allclose(array, array.T)


def check_Psi(
    result: FloatArray,
    expected_start: FloatArray,
    cutoff_index: int,
    expected_cutoff: FloatArray,
    expected_final: FloatArray,
):
    assert_is_symmetric(result[0, ...])

    # Mixed zeta elements: comparitively large atol because these are
    # expected to be zero
    assert np.isclose(result[0, 0, 1], expected_start[0, 1], rtol=1e-2, atol=0.4)
    assert np.isclose(result[0, 1, 2], expected_start[1, 2], rtol=1e-2, atol=0.4)

    # Diagonal elements
    assert_allclose(
        result[0, ...].diagonal(), expected_start.diagonal(), rtol=1e-2, atol=0.1
    )
    # RZ element
    assert np.isclose(result[0, 0, 2], expected_start[0, 2], rtol=1e-2, atol=0.1)

    assert_allclose(result[cutoff_index, ...], expected_cutoff, rtol=1e-2, atol=0.1)
    assert_allclose(result[-1, ...], expected_final, rtol=1.8e-2, atol=0.1)


@pytest.mark.parametrize(
    "generator",
    [
        pytest.param(simple, id="simple"),
        pytest.param(ne_dat_file, id="density-fit-file"),
        pytest.param(torbeam_file, id="torbeam-file"),
        pytest.param(npz_file, id="test-file"),
        # Following methods have errors of 30%, despite field being
        # idential to 1%, needs investigating
        # pytest.param(UDA_saved, id="UDA-saved-file"),
    ],
)
def test_integrated_O_mode(tmp_path, generator):
    """Golden answer test to check basic functionality using circular
    flux surfaces, mode flag set to 1."""

    kwargs_dict = generator(tmp_path)
    kwargs_dict["output_filename_suffix"] = "_Bpa0.10"
    kwargs_dict["figure_flag"] = False
    kwargs_dict["len_tau"] = 10
    kwargs_dict["output_path"] = tmp_path

    number_of_existing_npz_files = len(list(tmp_path.glob("*.npz")))

    beam_me_up(**kwargs_dict)

    assert len(list(tmp_path.glob("*.npz"))) == 4 + number_of_existing_npz_files

    with np.load(tmp_path / "data_output_Bpa0.10.npz") as f:
        output = dict(f)

    for key, value in EXPECTED_1.items():
        assert_allclose(output[key], value, rtol=1e-2, atol=1e-2, err_msg=key)

    K_magnitude = np.hypot(output["K_R_array"], output["K_Z_array"])
    assert K_magnitude.argmin() == CUTOFF_INDEX_1

    check_Psi(
        output["Psi_3D_output"],
        PSI_START_EXPECTED_1,
        CUTOFF_INDEX_1,
        PSI_CUTOFF_EXPECTED_1,
        PSI_FINAL_EXPECTED_1,
    )


# tests cannot run without assigning a unique generator to each test for reasons I don't understand - KR
@pytest.mark.parametrize(
    "generatorneg",
    [
        pytest.param(simple, id="simple"),
        pytest.param(ne_dat_file, id="density-fit-file"),
        pytest.param(torbeam_file, id="torbeam-file"),
        pytest.param(npz_file, id="test-file"),
        # Following methods have errors of 30%, despite field being
        # idential to 1%, needs investigating
        # pytest.param(UDA_saved, id="UDA-saved-file"),
    ],
)
def test_integrated_X_mode(tmp_path, generatorneg):
    """Golden answer test to check basic functionality using circular
    flux surfaces, mode flag set to -1."""

    kwargs_dict = generatorneg(tmp_path)
    kwargs_dict["output_filename_suffix"] = "_Bpa0.11"
    kwargs_dict["figure_flag"] = False
    kwargs_dict["len_tau"] = 10
    kwargs_dict["output_path"] = tmp_path
    kwargs_dict["mode_flag"] = -1

    number_of_existing_npz_files = len(list(tmp_path.glob("*.npz")))

    beam_me_up(**kwargs_dict)

    assert len(list(tmp_path.glob("*.npz"))) == 4 + number_of_existing_npz_files

    with np.load(tmp_path / "data_output_Bpa0.11.npz") as f:
        output = dict(f)

    for key, value in EXPECTED_NEG1.items():
        assert_allclose(output[key], value, rtol=1e-2, atol=1e-2, err_msg=key)

    K_magnitude = np.hypot(output["K_R_array"], output["K_Z_array"])
    assert K_magnitude.argmin() == CUTOFF_INDEX_NEG1

    check_Psi(
        output["Psi_3D_output"],
        PSI_START_EXPECTED_NEG1,
        CUTOFF_INDEX_NEG1,
        PSI_CUTOFF_EXPECTED_NEG1,
        PSI_FINAL_EXPECTED_NEG1,
    )


@pytest.mark.parametrize(
    "generator_rel",
    [
        pytest.param(simple, id="simple"),
        pytest.param(ne_dat_file, id="density-fit-file"),
        pytest.param(torbeam_file, id="torbeam-file"),
        pytest.param(npz_file, id="test-file"),
        # Following methods have errors of 30%, despite field being
        # idential to 1%, needs investigating
        # pytest.param(UDA_saved, id="UDA-saved-file"),
    ],
)
def test_relativistic_O_mode(tmp_path, generator_rel):
    """Golden answer test to check relativistic corrections. Mode flag = 1"""
    Te_fit = QuadraticFit(1.0, 10.0)

    kwargs_dict = generator_rel(tmp_path)
    kwargs_dict["output_filename_suffix"] = "_Bpa1.10"
    kwargs_dict["figure_flag"] = False
    kwargs_dict["len_tau"] = 10
    kwargs_dict["output_path"] = tmp_path
    kwargs_dict["relativistic_flag"] = True
    kwargs_dict["temperature_fit_method"] = Te_fit

    number_of_existing_npz_files = len(list(tmp_path.glob("*.npz")))
    print(kwargs_dict)
    beam_me_up(**kwargs_dict)

    assert len(list(tmp_path.glob("*.npz"))) == 4 + number_of_existing_npz_files

    with np.load(tmp_path / "data_output_Bpa1.10.npz") as f:
        output = dict(f)

    for key, value in EXPECTED_REL_1.items():
        assert_allclose(output[key], value, rtol=1e-2, atol=1e-2, err_msg=key)

    K_magnitude = np.hypot(output["K_R_array"], output["K_Z_array"])
    assert K_magnitude.argmin() == CUTOFF_INDEX_REL_1

    check_Psi(
        output["Psi_3D_output"],
        PSI_START_EXPECTED_REL_1,
        CUTOFF_INDEX_REL_1,
        PSI_CUTOFF_EXPECTED_REL_1,
        PSI_FINAL_EXPECTED_REL_1,
    )


@pytest.mark.parametrize(
    "generator_relneg",
    [
        pytest.param(simple, id="simple"),
        pytest.param(ne_dat_file, id="density-fit-file"),
        pytest.param(torbeam_file, id="torbeam-file"),
        pytest.param(npz_file, id="test-file"),
        # Following methods have errors of 30%, despite field being
        # idential to 1%, needs investigating
        # pytest.param(UDA_saved, id="UDA-saved-file"),
    ],
)
def test_relativistic_X_mode(tmp_path, generator_relneg):
    """Golden answer test to check relativistic corrections.Mode flag = -1"""
    Te_fit = QuadraticFit(1.0, 10.0)

    kwargs_dict = generator_relneg(tmp_path)
    kwargs_dict["output_filename_suffix"] = "_Bpa1.11"
    kwargs_dict["figure_flag"] = False
    kwargs_dict["len_tau"] = 10
    kwargs_dict["output_path"] = tmp_path
    kwargs_dict["relativistic_flag"] = True
    kwargs_dict["temperature_fit_method"] = Te_fit
    kwargs_dict["mode_flag"] = -1

    number_of_existing_npz_files = len(list(tmp_path.glob("*.npz")))
    print(kwargs_dict)
    beam_me_up(**kwargs_dict)

    assert len(list(tmp_path.glob("*.npz"))) == 4 + number_of_existing_npz_files

    with np.load(tmp_path / "data_output_Bpa1.11.npz") as f:
        output = dict(f)

    for key, value in EXPECTED_REL_NEG1.items():
        assert_allclose(output[key], value, rtol=1e-2, atol=1e-2, err_msg=key)

    K_magnitude = np.hypot(output["K_R_array"], output["K_Z_array"])
    assert K_magnitude.argmin() == CUTOFF_INDEX_REL_NEG1

    check_Psi(
        output["Psi_3D_output"],
        PSI_START_EXPECTED_REL_NEG1,
        CUTOFF_INDEX_REL_NEG1,
        PSI_CUTOFF_EXPECTED_REL_NEG1,
        PSI_FINAL_EXPECTED_REL_NEG1,
    )


@pytest.mark.parametrize(
    "generator_nullrel",
    [
        pytest.param(simple, id="simple"),
        pytest.param(ne_dat_file, id="density-fit-file"),
        pytest.param(torbeam_file, id="torbeam-file"),
        pytest.param(npz_file, id="test-file"),
        # Following methods have errors of 30%, despite field being
        # idential to 1%, needs investigating
        # pytest.param(UDA_saved, id="UDA-saved-file"),
    ],
)
def test_null_relativistic_O_mode(tmp_path, generator_nullrel):
    """Golden answer test to check relativistic corrections with zero temperature array.
    Output should be identical to nonrelativistic outputs. Mode_flag = 1."""

    Te_fit = QuadraticFit(1.0, 0.0)

    kwargs_dict = generator_nullrel(tmp_path)
    kwargs_dict["output_filename_suffix"] = "_Bpa2.10"
    kwargs_dict["figure_flag"] = False
    kwargs_dict["len_tau"] = 10
    kwargs_dict["output_path"] = tmp_path
    kwargs_dict["relativistic_flag"] = True
    kwargs_dict["temperature_fit_method"] = Te_fit

    number_of_existing_npz_files = len(list(tmp_path.glob("*.npz")))

    beam_me_up(**kwargs_dict)

    assert len(list(tmp_path.glob("*.npz"))) == 4 + number_of_existing_npz_files

    with np.load(tmp_path / "data_output_Bpa2.10.npz") as f:
        output = dict(f)

    for key, value in EXPECTED_1.items():
        assert_allclose(output[key], value, rtol=1e-2, atol=1e-2, err_msg=key)

    K_magnitude = np.hypot(output["K_R_array"], output["K_Z_array"])
    assert K_magnitude.argmin() == CUTOFF_INDEX_1

    check_Psi(
        output["Psi_3D_output"],
        PSI_START_EXPECTED_1,
        CUTOFF_INDEX_1,
        PSI_CUTOFF_EXPECTED_1,
        PSI_FINAL_EXPECTED_1,
    )


@pytest.mark.parametrize(
    "generator_nullrelneg",
    [
        pytest.param(simple, id="simple"),
        pytest.param(ne_dat_file, id="density-fit-file"),
        pytest.param(torbeam_file, id="torbeam-file"),
        pytest.param(npz_file, id="test-file"),
        # Following methods have errors of 30%, despite field being
        # idential to 1%, needs investigating
        # pytest.param(UDA_saved, id="UDA-saved-file"),
    ],
)
def test_null_relativistic_X_mode(tmp_path, generator_nullrelneg):
    """Golden answer test to check relativistic corrections with zero temperature array.
    Output should be identical to nonrelativistic outputs. Mode_flag = -1."""

    Te_fit = QuadraticFit(1.0, 0.0)

    kwargs_dict = generator_nullrelneg(tmp_path)
    kwargs_dict["output_filename_suffix"] = "_Bpa2.11"
    kwargs_dict["figure_flag"] = False
    kwargs_dict["len_tau"] = 10
    kwargs_dict["output_path"] = tmp_path
    kwargs_dict["relativistic_flag"] = True
    kwargs_dict["temperature_fit_method"] = Te_fit
    kwargs_dict["mode_flag"] = -1

    number_of_existing_npz_files = len(list(tmp_path.glob("*.npz")))

    beam_me_up(**kwargs_dict)

    assert len(list(tmp_path.glob("*.npz"))) == 4 + number_of_existing_npz_files

    with np.load(tmp_path / "data_output_Bpa2.11.npz") as f:
        output = dict(f)

    for key, value in EXPECTED_NEG1.items():
        assert_allclose(output[key], value, rtol=1e-2, atol=1e-2, err_msg=key)

    K_magnitude = np.hypot(output["K_R_array"], output["K_Z_array"])
    assert K_magnitude.argmin() == CUTOFF_INDEX_NEG1

    check_Psi(
        output["Psi_3D_output"],
        PSI_START_EXPECTED_NEG1,
        CUTOFF_INDEX_NEG1,
        PSI_CUTOFF_EXPECTED_NEG1,
        PSI_FINAL_EXPECTED_NEG1,
    )


@pytest.mark.parametrize(
    "generator",
    [
        pytest.param(ne_dat_file, id="density-fit-file"),
        pytest.param(torbeam_file, id="torbeam-file"),
        pytest.param(omfit_json, id="omfit-file"),
        pytest.param(npz_file, id="test-file"),
        pytest.param(npz_notime_file, id="test_notime-file"),
        pytest.param(UDA_saved, id="UDA-saved-file"),
        pytest.param(UDA_saved_MAST_U, id="UDA-saved-MAST-U-file"),
    ],
)
def test_create_magnetic_geometry(tmp_path, generator):
    kwargs_dict = generator(tmp_path)

    field = create_magnetic_geometry(**kwargs_dict)
    field_golden = create_magnetic_geometry(**simple(tmp_path))

    R_axis = kwargs_dict["R_axis"]
    minor_radius = kwargs_dict["minor_radius_a"]
    R = np.linspace(R_axis + 0.1, R_axis + minor_radius, 10)
    Z = np.linspace(0.1, kwargs_dict["minor_radius_a"], 10)

    assert_allclose(
        field.B_R(R_axis, Z), field_golden.B_R(R_axis, Z), rtol=1e-3, atol=1e-3
    )
    assert_allclose(field.B_T(R, 0.0), field_golden.B_T(R, 0.0), rtol=1e-3, atol=1e-3)
    assert_allclose(field.B_Z(R, 0.0), field_golden.B_Z(R, 0.0), rtol=1e-3, atol=1e-3)
    assert_allclose(
        field.poloidal_flux(R, 0.0),
        field_golden.poloidal_flux(R, 0.0),
        rtol=1e-3,
        atol=1e-3,
    )


def test_EFIT_geometry(tmp_path):
    kwargs_dict = UDA_saved(tmp_path)
    field = create_magnetic_geometry(**kwargs_dict)
    field_golden = create_magnetic_geometry(**simple(tmp_path))

    R_axis = kwargs_dict["R_axis"]
    minor_radius_a = kwargs_dict["minor_radius_a"]
    B_p_a = kwargs_dict["B_p_a"]

    # Include buffer for gradient near edge
    width = minor_radius_a + 0.05
    # Different grid sizes to capture transpose errors
    R = np.linspace(R_axis - width, R_axis + width, 99)
    Z = np.linspace(-width, width, 101)
    R_grid, Z_grid = np.meshgrid(R, Z, indexing="ij")
    B_R = field.B_R(R_grid, Z_grid)
    B_Z = field.B_Z(R_grid, Z_grid)

    psi = field.poloidal_flux(R_grid, Z_grid)
    grad_psi = np.gradient(psi, R, Z)
    calculated_B_R = grad_psi[1] * B_p_a / R_grid
    calculated_B_Z = -grad_psi[0] * B_p_a / R_grid

    mask = (psi > 0.1) & (psi <= 1)

    assert_allclose(B_R[mask], calculated_B_R[mask], 2e-3, 2e-3)
    assert_allclose(B_Z[mask], calculated_B_Z[mask], 2e-3, 2e-3)

    assert_allclose(field.B_T(R_grid, Z_grid), field_golden.B_T(R_grid, Z_grid))

    # Check that poloidal field rotates in the correct direction
    total_sign = np.sign(B_R) * np.sign(B_Z)
    assert total_sign[0, 0] == -1, "Top left"
    assert total_sign[0, -1] == 1, "Top right"
    assert total_sign[-1, 0] == 1, "Bottom left"
    assert total_sign[-1, -1] == -1, "Bottom right"


@pytest.mark.parametrize(
    "generator",
    [
        pytest.param(simple, id="simple"),
        pytest.param(ne_dat_file, id="density-fit-file"),
        pytest.param(torbeam_file, id="torbeam-file"),
        pytest.param(omfit_json, id="omfit-file"),
        pytest.param(npz_file, id="test-file"),
        pytest.param(npz_notime_file, id="test_notime-file"),
        # pytest.param(UDA_saved, id="UDA-saved-file"),
        # pytest.param(UDA_saved_MAST_U, id="UDA-saved-MAST-U-file"),
    ],
)
def test_launch_golden_answer(tmp_path, generator):
    args = generator(tmp_path)
    field = create_magnetic_geometry(**args)
    ne_data_path = args.get("ne_data_path", None)
    if ne_data_path:
        ne_filename = ne_data_path / "ne.dat"
        parameters = [ne_filename, 5, 0]
    else:
        parameters = None
        ne_filename = None

    density_fit = make_density_fit(
        args["density_fit_method"],
        args["poloidal_flux_zero_density"],
        parameters,
        ne_filename,
    )

    launch_angular_frequency = freq_GHz_to_angular_frequency(args["launch_freq_GHz"])

    hamiltonian = Hamiltonian(
        field,
        launch_angular_frequency,
        mode_flag=args["mode_flag"],
        density_fit=density_fit,
        delta_R=-1e-3,
        delta_Z=1e-3,
        delta_K_R=0.1,
        delta_K_zeta=0.1,
        delta_K_Z=0.1,
    )
    (
        K_initial,
        initial_position,
        launch_K,
        Psi_3D_lab_initial,
        Psi_3D_lab_launch,
        Psi_3D_lab_entry,
        Psi_3D_lab_entry_cartersian,
        distance_from_launch_to_entry,
    ) = launch_beam(
        field=field,
        hamiltonian=hamiltonian,
        toroidal_launch_angle_Torbeam=args["toroidal_launch_angle_Torbeam"],
        poloidal_launch_angle_Torbeam=args["poloidal_launch_angle_Torbeam"],
        launch_beam_width=args["launch_beam_width"],
        launch_beam_curvature=args["launch_beam_curvature"],
        launch_position=args["launch_position"],
        launch_angular_frequency=launch_angular_frequency,
        mode_flag=args["mode_flag"],
        vacuum_propagation_flag=args["vacuum_propagation_flag"],
        Psi_BC_flag=args["Psi_BC_flag"],
        poloidal_flux_enter=args["poloidal_flux_enter"],
    )

    expected_K_initial = np.array([-1146.4000699962507, -0.0, -120.4915026654739])
    expected_initial_position = np.array([1.99382564, -0.0, -0.07804514])
    expected_launch_K = np.array([-1146.40007, -0.0, -120.49150267])
    # Note that R-zeta, Z-zeta terms are not actually zero, but should be
    expected_Psi_3D_lab_initial = np.array(
        [
            [-2.473578e03 + 1.195578e01j, 0.0 + 0.0j, 3.405239e02 - 1.137526e02j],
            [0.0 + 0.0j, 4.245892e03 + 4.350010e03j, 0.0 + 0.0j],
            [3.405239e02 - 1.137526e02j, 0.0 + 0.0j, 4.257764e02 + 1.082292e03j],
        ]
    )
    expected_Psi_3D_lab_launch = np.array(
        [
            [-3.1486979 + 13.65774954j, 0.0 + 0.0j, 29.9578594 - 129.94480676j],
            [0.0 + 0.0j, 1037.08121046 + 8365.71125j, 0.0 + 0.0j],
            [29.9578594 - 129.94480676j, 0.0 + 0.0j, -285.02999262 + 1236.34225046j],
        ]
    )
    expected_Psi_3D_lab_entry = np.array(
        [
            [5.38751589 + 11.95597601j, 0.0 + 0.0j, -51.25878964 - 113.75351313j],
            [0.0 + 0.0j, 4245.89207422 + 4350.0100231j, 0.0 + 0.0j],
            [-51.25878964 - 113.75351313j, 0.0 + 0.0j, 487.69480611 + 1082.29238188j],
        ]
    )
    expected_Psi_3D_lab_entry_cartersian = np.array(
        [
            [5.38751589 + 11.95597601j, 0.0 + 0.0j, -51.25878964 - 113.75351313j],
            [0.0 + 0.0j, 493.082322 + 1094.24835789j, 0.0 + 0.0j],
            [-51.25878964 - 113.75351313j, 0.0 + 0.0j, 487.69480611 + 1082.29238188j],
        ]
    )
    expected_distance_from_launch_to_entry = 0.5964417281350541
    expected_Psi_3D = np.array(
        [
            [-3.1486979 + 13.65774954j, 0.0 + 0.0j, 29.9578594 - 129.94480676j],
            [0.0 + 0.0j, 1037.08121046 + 8365.71125j, 0.0 + 0.0j],
            [29.9578594 - 129.94480676j, 0.0 + 0.0j, -285.02999262 + 1236.34225046j],
        ]
    )

    tol = 1e-3
    assert_allclose(Psi_3D_lab_launch, expected_Psi_3D, tol, tol)
    assert_allclose(K_initial, expected_K_initial, tol, tol)
    assert_allclose(initial_position, expected_initial_position, tol, tol)
    assert_allclose(launch_K, expected_launch_K, tol, tol)
    assert_allclose(Psi_3D_lab_launch, expected_Psi_3D_lab_launch, tol, tol)
    assert_allclose(Psi_3D_lab_entry, expected_Psi_3D_lab_entry, tol, tol)
    assert_allclose(
        Psi_3D_lab_entry_cartersian, expected_Psi_3D_lab_entry_cartersian, tol, tol
    )
    assert_allclose(
        distance_from_launch_to_entry,
        expected_distance_from_launch_to_entry,
        tol,
        tol,
    )
    assert_allclose(launch_K, expected_launch_K, tol, tol)
    assert_allclose(Psi_3D_lab_launch, expected_Psi_3D_lab_launch, tol, tol)
    assert_allclose(Psi_3D_lab_entry, expected_Psi_3D_lab_entry, tol, tol)
    assert_allclose(
        Psi_3D_lab_entry_cartersian, expected_Psi_3D_lab_entry_cartersian, tol, tol
    )
    # Larger atol due to some small values
    assert_allclose(Psi_3D_lab_initial, expected_Psi_3D_lab_initial, tol, atol=0.4)


def launch_parameters(start_point, end_point_poloidal_coords):
    kwargs_dict = simple(None)
    rho_end, theta_end, zeta_end = end_point_poloidal_coords
    R_axis = kwargs_dict["R_axis"]
    minor_radius = kwargs_dict["minor_radius_a"]

    R_end = R_axis + rho_end * minor_radius * np.cos(theta_end)
    Z_end = rho_end * minor_radius * np.sin(theta_end)
    X_end = R_end * np.cos(zeta_end)
    Y_end = R_end * np.sin(zeta_end)

    X_start, Y_start, Z_start = start_point
    phi_t = np.arctan2(Y_end - Y_start, X_end - X_start) - np.pi
    poloidal_length = np.sqrt((Y_end - Y_start) ** 2 + (X_end - X_start) ** 2)
    phi_p = np.arctan2(Z_end - Z_start, poloidal_length)
    expected_entry_cartesian = (R_end, zeta_end, Z_end)
    return start_point, -phi_p, phi_t, expected_entry_cartesian


@pytest.mark.parametrize(
    "generator",
    [
        pytest.param(simple, id="simple"),
        pytest.param(ne_dat_file, id="density-fit-file"),
        pytest.param(torbeam_file, id="torbeam-file"),
        pytest.param(npz_file, id="test-file"),
        pytest.param(UDA_saved, id="UDA-saved-file"),
    ],
)
@pytest.mark.parametrize(
    (
        "launch_position",
        "poloidal_launch_angle",
        "toroidal_launch_angle",
        "expected_entry",
    ),
    (
        pytest.param([2.5, 0, 0], 0, 0, [2.0, 0.0, 0], id="outboard-midplane"),
        pytest.param([0.5, 0, 0], 0, np.pi, [1.0, 0, 0], id="inboard-midplane"),
        pytest.param([1.5, 0, 1], np.pi / 2, 0, [1.5, 0, 0.5], id="top"),
        pytest.param([1.5, 0, -1], -np.pi / 2, 0, [1.5, 0, -0.5], id="bottom"),
        pytest.param(
            [2, 0, 0.5],
            np.pi / 4,
            0,
            [1.5 + np.sqrt(0.5) / 2, 0, np.sqrt(0.5) / 2],
            id="top-right",
        ),
        pytest.param(
            [2, 0, -0.5],
            -np.pi / 4,
            0,
            [1.5 + np.sqrt(0.5) / 2, 0, -np.sqrt(0.5) / 2],
            id="bottom-right",
        ),
        pytest.param(
            *launch_parameters((2.5, 0.0, -0.1), (1, np.pi / 4, np.pi / 8)),
            id="steep-angle",
        ),
    ),
)
def test_find_entry_point(
    tmp_path,
    generator,
    launch_position,
    poloidal_launch_angle,
    toroidal_launch_angle,
    expected_entry,
):
    args = generator(tmp_path)
    field = create_magnetic_geometry(**args)
    poloidal_flux_enter = args["poloidal_flux_enter"]

    entry_position = find_entry_point(
        launch_position,
        poloidal_launch_angle,
        toroidal_launch_angle,
        poloidal_flux_enter,
        field,
    )

    assert_allclose(entry_position, expected_entry, 1e-6, 1e-6)


def test_find_entry_point_bug(tmp_path):
    args = simple(tmp_path)
    field = create_magnetic_geometry(**args)
    R, zeta, Z = find_entry_point(
        launch_position=[2.3, 0, -0.1],
        poloidal_launch_angle=np.deg2rad(-1),
        toroidal_launch_angle=0.0,
        poloidal_flux_enter=1.0,
        field=field,
    )

    assert np.isclose(zeta, 0.0)


@pytest.mark.parametrize(
    "generator",
    [pytest.param(simple, id="simple")],
)
def test_quick_run(tmp_path, generator):
    """Golden answer test to check basic functionality using circular
    flux surfaces with quick_run=True"""

    kwargs_dict = generator(tmp_path)
    kwargs_dict["output_filename_suffix"] = "_Bpa0.10"
    kwargs_dict["figure_flag"] = False
    kwargs_dict["len_tau"] = 10
    kwargs_dict["output_path"] = tmp_path
    kwargs_dict["quick_run"] = True

    q_R_cutoff_expected = 1.694
    q_Z_cutoff_expected = -0.18097
    K_norm_expected = 561.52
    poloidal_flux_cutoff_expected = 0.53061
    theta_m_cutoff_expected = 0.132552

    cutoff = beam_me_up(**kwargs_dict)

    assert np.isclose(cutoff.q_R, q_R_cutoff_expected)
    assert np.isclose(cutoff.q_Z, q_Z_cutoff_expected)
    assert np.isclose(cutoff.K_norm_min, K_norm_expected)
    assert np.isclose(cutoff.poloidal_flux, poloidal_flux_cutoff_expected)
    assert np.isclose(cutoff.theta_m, theta_m_cutoff_expected)
