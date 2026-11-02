from scotty.beam_me_up import beam_me_up, create_magnetic_geometry, make_fit
from scotty.init_bruv import get_parameters_for_Scotty
from scotty.torbeam import Torbeam
from scotty.geometry import CircularFluxSurfaceField, MagneticField_Cylindrical
from scotty.fun_general import freq_GHz_to_angular_frequency
from scotty.profile_fit import QuadraticFit
from scotty.hamiltonian import initialise_hamiltonians
from scotty.launch import (
    find_auto_delta_signs,
    find_plasma_entry_position,
    find_plasma_entry_parameters,
)
from scotty.typing import FloatArray

import json
import numpy as np
from numpy.testing import assert_allclose
import pathlib
import pytest
import xarray as xr
from typing import Optional, cast

# Print more of arrays in failed tests
np.set_printoptions(linewidth=120, threshold=100)


def synthetic_parameters() -> dict:
    """The DBS_synthetic preset, with the v3 settings the expected values were
    generated with: no interpolation smoothing, and v3's default spatial
    finite-difference steps (v4's default is 1e-5)"""
    kwargs_dict = get_parameters_for_Scotty("DBS_synthetic")
    kwargs_dict["interp_smoothing"] = 0
    kwargs_dict["delta_R"] = -1e-4
    kwargs_dict["delta_Z"] = 1e-4
    return kwargs_dict


def make_field(kwargs_dict: dict):
    """Call create_magnetic_geometry with the v3 default interpolation
    settings (quintic, no smoothing)"""
    kwargs_dict = kwargs_dict.copy()
    return create_magnetic_geometry(
        interp_order_str="quintic",
        interp_order_int=5,
        interp_smoothing=kwargs_dict.pop("interp_smoothing", 0),
        magnetic_data_path=kwargs_dict.pop("magnetic_data_path", pathlib.Path(".")),
        **kwargs_dict,
    )


# For circular flux surfaces, O-mode mode_flag=1, X-mode mode_flag=-1

# The expected values are v3's (upstream/master, feb9499) on the same cases,
# with find_B_method="analytical" (CircularFluxSurfaceField), 2026-11-02

# Expected values for nonrelativistic simple golden, mode_flag = 1

CUTOFF_INDEX_1 = 5

PSI_START_EXPECTED_1 = np.array(
    [
        [
            -4955.49774 + 11.9586918j,
            -0.0643544005 - 0.0659383544j,
            732.657352 - 113.768878j,
        ],
        [
            -0.0643544005 - 0.0659383544j,
            4245.88293 + 4350.38679j,
            0.0101690972 + 0.010419389j,
        ],
        [
            732.657352 - 113.768878j,
            0.0101690972 + 0.010419389j,
            363.783833 + 1082.33902j,
        ],
    ]
)
PSI_CUTOFF_EXPECTED_1 = np.array(
    [
        [
            -6216.99499 + 1107.6683j,
            -175.140574 - 86.1774989j,
            -253.259902 - 638.806752j,
        ],
        [
            -175.140574 - 86.1774989j,
            4124.23367 + 3239.22157j,
            -50.7647998 - 69.5850083j,
        ],
        [
            -253.259902 - 638.806752j,
            -50.7647998 - 69.5850083j,
            2360.90913 + 370.605126j,
        ],
    ]
)
PSI_FINAL_EXPECTED_1 = np.array(
    [
        [
            9353.79464 + 2946.2615j,
            37.8091536 + 45.3504337j,
            -130.889383 + 933.169434j,
        ],
        [
            37.8091536 + 45.3504337j,
            3961.09542 + 2715.27774j,
            -29.1793326 - 13.672076j,
        ],
        [
            -130.889383 + 933.169434j,
            -29.1793326 - 13.672076j,
            3321.13793 + 293.366788j,
        ],
    ]
)


EXPECTED_1 = {
    "solver_output/q_R_output": np.array(
        [
            1.99387212,
            1.91867246,
            1.86531618,
            1.82693842,
            1.7999295,
            1.7901642,
            1.78254356,
            1.77423261,
            1.77549019,
            1.78804407,
        ]
    ),
    "solver_output/q_Z_output": np.array(
        [
            -0.0780402519,
            -0.0899700029,
            -0.106506272,
            -0.127397785,
            -0.152741683,
            -0.167084626,
            -0.182998261,
            -0.219119196,
            -0.262862761,
            -0.317399807,
        ]
    ),
    "solver_output/q_zeta_output": np.array(
        [
            3.64302277e-17,
            -0.000333582376,
            -0.00108631702,
            -0.00205028427,
            -0.00311777841,
            -0.00366494674,
            -0.00422401077,
            -0.005311707,
            -0.00631162572,
            -0.00710860701,
        ]
    ),
    "solver_output/K_R_output": np.array(
        [
            -1146.40007,
            -800.675087,
            -574.797412,
            -411.568972,
            -280.82811,
            -222.182355,
            -164.302969,
            -48.1293634,
            81.6285072,
            244.095056,
        ]
    ),
    "solver_output/K_Z_output": np.array(
        [
            -120.491503,
            -183.759751,
            -240.206593,
            -295.297114,
            -353.696981,
            -385.490088,
            -420.864775,
            -504.80429,
            -618.66801,
            -786.183905,
        ]
    ),
    "solver_output/K_zeta_output": np.array(8.327130896577119e-14),
    "solver_output/tau": np.array(
        [
            0,
            52.0411325,
            104.082265,
            156.123397,
            208.16453,
            233.96946,
            260.205662,
            312.246795,
            364.287927,
            416.32906,
        ]
    ),
    "analysis/B_magnitude": np.array(
        [
            0.756057183,
            0.784652628,
            0.806478672,
            0.823064978,
            0.835252042,
            0.839788673,
            0.843399154,
            0.847516807,
            0.847283454,
            0.841982198,
        ]
    ),
    "analysis/normalised_omega_ce": np.array(
        [
            0.3847986,
            0.399352377,
            0.410460837,
            0.41890251,
            0.425105169,
            0.427414107,
            0.429251677,
            0.431347375,
            0.431228608,
            0.428530511,
        ]
    ),
    "analysis/normalised_omega_pe": np.array(
        [
            0.000616658561,
            0.701733681,
            0.84165899,
            0.898588841,
            0.920381914,
            0.92283304,
            0.920315767,
            0.898360122,
            0.8410945,
            0.700256383,
        ]
    ),
}

# Expected values for nonrelativistic simple golden, mode_flag = -1

CUTOFF_INDEX_NEG1 = 5

PSI_START_EXPECTED_NEG1 = np.array(
    [
        [
            -5825.35284 + 11.9542186j,
            0.0643578786 + 0.0659419182j,
            870.094784 - 113.747245j,
        ],
        [
            0.0643578786 + 0.0659419182j,
            4245.88293 + 4350.38679j,
            -0.0101696468 - 0.0104199522j,
        ],
        [
            870.094784 - 113.747245j,
            -0.0101696468 - 0.0104199522j,
            342.068668 + 1082.33229j,
        ],
    ]
)
# Psi at the cut-off is steep (Psi_RR ~ -2.6e5, dPsi_RR/dtau ~ 6e3), and at
# the default rtol = 1e-3 v3 is 0.35% and v4 0.9% from the converged value,
# so it is checked on its own at rtol = 1e-6 (test_X_mode_cutoff_Psi), against
# v3 (upstream/master feb9499, find_B_method="analytical") at rtol = 1e-6,
# len_tau = 10; v3's Te = 0 relativistic run gives the same values
TAU_CUTOFF_EXPECTED_NEG1_RTOL_1E_6 = 95.5291896
PSI_CUTOFF_EXPECTED_NEG1_RTOL_1E_6 = np.array(
    [
        [
            -256668.47 + 62324.0167j,
            1387.63788 + 415.931269j,
            40418.182 - 16269.5368j,
        ],
        [
            1387.63788 + 415.931269j,
            3928.8549 + 2641.77702j,
            -204.652169 - 2.47675584j,
        ],
        [
            40418.182 - 16269.5368j,
            -204.652169 - 2.47675584j,
            -3586.42783 + 4251.38834j,
        ],
    ]
)
PSI_FINAL_EXPECTED_NEG1 = np.array(
    [
        [
            7878.94446 + 158.034759j,
            4.97014272 + 8.67221797j,
            -292.694185 + 193.732903j,
        ],
        [
            4.97014272 + 8.67221797j,
            3482.5037 + 1816.34511j,
            32.5956737 + 24.5841758j,
        ],
        [
            -292.694185 + 193.732903j,
            32.5956737 + 24.5841758j,
            3305.56964 + 240.29595j,
        ],
    ]
)


EXPECTED_NEG1 = {
    "solver_output/q_R_output": np.array(
        [
            1.99387212,
            1.96059083,
            1.93375615,
            1.91382316,
            1.90187889,
            1.89921798,
            1.89892422,
            1.90491757,
            1.91879627,
            1.93940959,
        ]
    ),
    "solver_output/q_Z_output": np.array(
        [
            -0.0780402519,
            -0.0824297705,
            -0.0879817143,
            -0.0949140964,
            -0.103611144,
            -0.108884544,
            -0.114513539,
            -0.127901793,
            -0.143703861,
            -0.161754612,
        ]
    ),
    "solver_output/q_zeta_output": np.array(
        [
            3.64302277e-17,
            7.7115709e-05,
            0.000289277465,
            0.000613135219,
            0.00101889766,
            0.0012419014,
            0.00145442139,
            0.00185586663,
            0.00217350912,
            0.00238006514,
        ]
    ),
    "solver_output/K_R_output": np.array(
        [
            -1146.40007,
            -943.487694,
            -738.269903,
            -507.308525,
            -237.498694,
            -89.3380285,
            52.0238048,
            319.867384,
            546.641157,
            745.322637,
        ]
    ),
    "solver_output/K_Z_output": np.array(
        [
            -120.491503,
            -154.448341,
            -192.960528,
            -241.465278,
            -304.748976,
            -342.44173,
            -380.386699,
            -458.052762,
            -530.676838,
            -600.310963,
        ]
    ),
    "solver_output/K_zeta_output": np.array(8.327130896577119e-14),
    "solver_output/tau": np.array(
        [
            0,
            21.1774728,
            42.3549457,
            63.5324185,
            84.7098914,
            95.5371374,
            105.887364,
            127.064837,
            148.24231,
            169.419783,
        ]
    ),
    "analysis/B_magnitude": np.array(
        [
            0.756057183,
            0.768418283,
            0.778725431,
            0.786592024,
            0.791406012,
            0.792498899,
            0.792637606,
            0.790270307,
            0.784799186,
            0.77681528,
        ]
    ),
    "analysis/normalised_omega_ce": np.array(
        [
            0.3847986,
            0.391089836,
            0.396335704,
            0.400339441,
            0.402789541,
            0.403345771,
            0.403416366,
            0.402211519,
            0.399426968,
            0.395363524,
        ]
    ),
    "analysis/normalised_omega_pe": np.array(
        [
            0.000616658561,
            0.498433924,
            0.641517632,
            0.715401529,
            0.748328729,
            0.752275896,
            0.748281731,
            0.715148688,
            0.640704682,
            0.496128757,
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
            -4955.49064 + 11.9564542j,
            -0.0194391357 - 0.0199175909j,
            732.661106 - 113.758062j,
        ],
        [
            -0.0194391357 - 0.0199175909j,
            4245.88293 + 4350.38679j,
            0.00307171629 + 0.0031473204j,
        ],
        [
            732.661106 - 113.758062j,
            0.00307171629 + 0.0031473204j,
            363.782469 + 1082.33566j,
        ],
    ]
)
PSI_CUTOFF_EXPECTED_REL_1 = np.array(
    [
        [
            -4191.03883 + 770.914021j,
            -143.536195 - 71.3057054j,
            -460.036706 - 489.471829j,
        ],
        [
            -143.536195 - 71.3057054j,
            4104.72049 + 3161.87284j,
            -58.4130076 - 70.0971102j,
        ],
        [
            -460.036706 - 489.471829j,
            -58.4130076 - 70.0971102j,
            2307.5308 + 315.952416j,
        ],
    ]
)
# The final point sits at tau_leave, which at the default rtol = 1e-3 is
# 2.5e-3 from v3's, so Psi there is 2.95% from v3 (check_Psi allows 1.8%);
# at rtol = 1e-6 in both codes they are 0.26% apart. So it is checked on its
# own at rtol = 1e-6 (test_relativistic_O_mode_final_Psi), against v3
# (upstream/master feb9499, find_B_method="analytical") at rtol = 1e-6,
# len_tau = 10
TAU_LEAVE_EXPECTED_REL_1_RTOL_1E_6 = 448.836518
PSI_FINAL_EXPECTED_REL_1_RTOL_1E_6 = np.array(
    [
        [
            2707.42091 + 13246.4581j,
            -99.2974494 + 166.075484j,
            -1947.64026 + 2713.01707j,
        ],
        [
            -99.2974494 + 166.075484j,
            3912.83486 + 2602.76635j,
            -62.2177954 + 6.18677245j,
        ],
        [
            -1947.64026 + 2713.01707j,
            -62.2177954 + 6.18677245j,
            2893.1764 + 555.952792j,
        ],
    ]
)


EXPECTED_REL_1 = {
    "solver_output/q_R_output": np.array(
        [
            1.99387212,
            1.9135442,
            1.85678651,
            1.81536467,
            1.78519174,
            1.77363543,
            1.76428835,
            1.75198388,
            1.74867752,
            1.75604055,
        ]
    ),
    "solver_output/q_Z_output": np.array(
        [
            -0.0780402519,
            -0.0910440689,
            -0.109125262,
            -0.131876825,
            -0.159349339,
            -0.174999433,
            -0.192008282,
            -0.230907021,
            -0.278033874,
            -0.337117394,
        ]
    ),
    "solver_output/q_zeta_output": np.array(
        [
            3.64302277e-17,
            -0.000375041028,
            -0.00120073994,
            -0.002242782,
            -0.00339163221,
            -0.00398552171,
            -0.00458186126,
            -0.0057580247,
            -0.00685042149,
            -0.00773851539,
        ]
    ),
    "solver_output/K_R_output": np.array(
        [
            -1146.40007,
            -788.991072,
            -569.992319,
            -417.925476,
            -299.784065,
            -247.182619,
            -196.69178,
            -95.1732822,
            17.957828,
            161.206655,
        ]
    ),
    "solver_output/K_Z_output": np.array(
        [
            -120.491503,
            -186.527638,
            -243.209195,
            -297.626044,
            -354.878927,
            -386.35761,
            -420.800255,
            -504.021578,
            -619.172751,
            -794.120485,
        ]
    ),
    "solver_output/K_zeta_output": np.array(8.327130896577119e-14),
    "solver_output/tau": np.array(
        [
            0,
            56.0970628,
            112.194126,
            168.291188,
            224.388251,
            252.469476,
            280.485314,
            336.582377,
            392.67944,
            448.776503,
        ]
    ),
    "analysis/B_magnitude": np.array(
        [
            0.756057183,
            0.786691848,
            0.81009332,
            0.828208969,
            0.8420375,
            0.847503199,
            0.852013818,
            0.858170185,
            0.8601749,
            0.857250447,
        ]
    ),
    "analysis/normalised_omega_ce": np.array(
        [
            0.384798594,
            0.391099851,
            0.398838747,
            0.405997861,
            0.412073895,
            0.414667522,
            0.416955338,
            0.4206828,
            0.423490254,
            0.426165676,
        ]
    ),
    "analysis/normalised_omega_pe": np.array(
        [
            0.000616658555,
            0.711067949,
            0.843490231,
            0.895797944,
            0.915528697,
            0.917765179,
            0.91554896,
            0.895864895,
            0.843652426,
            0.711490074,
        ]
    ),
}

# For mode_flag = -1:

CUTOFF_INDEX_REL_NEG1 = 5

PSI_START_EXPECTED_REL_NEG1 = np.array(
    [
        [
            -5825.36055 + 11.9564542j,
            0.0194428165 + 0.0199213623j,
            870.091141 - 113.758062j,
        ],
        [
            0.0194428165 + 0.0199213623j,
            4245.88293 + 4350.38679j,
            -0.00307229792 - 0.00314791634j,
        ],
        [
            870.091141 - 113.758062j,
            -0.00307229792 - 0.00314791634j,
            342.070011 + 1082.33566j,
        ],
    ]
)
# The cut-off point is inserted where |K| is smallest, and v3 and v4 find that
# minimum differently (v3: argmin on a fine grid; v4: minimize_scalar), 0.02%
# apart in tau. Psi is steep there (dPsi_RR/dtau ~ 5e4), so Psi at the
# cut-off is compared with v3's Psi at v4's cut-off tau, interpolated from
# this table. Table: v3 (upstream/master feb9499, find_B_method="analytical")
# on this case with len_tau = 8001, cubic spline of each component in tau; it
# reproduces the fine grid to 8e-5 between 102.6 and 103.5.
# (Bakso, 2026-10-30; regenerated on the analytical field, 2026-11-02.)
TAU_CUTOFF_EXPECTED_REL_NEG1 = 103.0875  # v3, len_tau = 8001
PSI_CUTOFF_TABLE_TAU_REL_NEG1 = np.array(
    [102.6, 102.75, 102.9, 103.05, 103.2, 103.35, 103.5]
)
_PSI_CUTOFF_TABLE_REL_NEG1 = {  # (row, col): values at the table's tau
    (0, 0): [
        -173195.1 + 34953.161j,
        -180056.6 + 37949.856j,
        -187455.94 + 41359.859j,
        -195452.52 + 45256.344j,
        -204110.94 + 49720.827j,
        -213500.94 + 54843.165j,
        -223679.33 + 60773.816j,
    ],
    (0, 1): [
        1005.5665 + 327.03969j,
        1048.739 + 330.52901j,
        1095.7808 + 333.3824j,
        1147.1947 + 335.38952j,
        1203.5321 + 336.31165j,
        1265.3938 + 335.88172j,
        1333.4801 + 333.48888j,
    ],
    (0, 2): [
        26856.26 - 10124.993j,
        27864.51 - 10781.54j,
        28946.9 - 11517.805j,
        30110.887 - 12346.949j,
        31364.524 - 13283.591j,
        32716.462 - 14343.807j,
        34171.691 - 15552.923j,
    ],
    (1, 1): [
        3886.9384 + 2540.8819j,
        3885.7032 + 2538.4063j,
        3884.4461 + 2535.926j,
        3883.1639 + 2533.4408j,
        3881.8532 + 2530.9504j,
        3880.5101 + 2528.4544j,
        3879.1296 + 2525.9542j,
    ],
    (1, 2): [
        -136.04697 + 6.8528069j,
        -142.92403 + 7.4713581j,
        -150.39677 + 8.2932636j,
        -158.53913 + 9.3651787j,
        -167.43207 + 10.739554j,
        -177.16355 + 12.474632j,
        -187.82819 + 14.685719j,
    ],
    (2, 2): [
        -1454.3143 + 2936.8252j,
        -1593.8979 + 3066.6654j,
        -1743.101 + 3211.0942j,
        -1902.7541 + 3372.3931j,
        -2073.7432 + 3553.0901j,
        -2257.0099 + 3755.9598j,
        -2452.6707 + 3985.1432j,
    ],
}


def psi_cutoff_expected_rel_neg1(tau_cutoff: float) -> np.ndarray:
    """v3's Psi (lab frame, cylindrical) at the given cut-off tau"""
    from scipy.interpolate import CubicSpline

    Psi = np.zeros((3, 3), dtype=complex)
    for (a, b), values in _PSI_CUTOFF_TABLE_REL_NEG1.items():
        values = np.array(values)
        Psi[a, b] = Psi[b, a] = CubicSpline(PSI_CUTOFF_TABLE_TAU_REL_NEG1, values.real)(
            tau_cutoff
        ) + 1j * CubicSpline(PSI_CUTOFF_TABLE_TAU_REL_NEG1, values.imag)(tau_cutoff)
    return Psi


PSI_FINAL_EXPECTED_REL_NEG1 = np.array(
    [
        [
            7743.65838 + 162.669792j,
            5.33242612 + 8.76253946j,
            -337.39782 + 187.765058j,
        ],
        [
            5.33242612 + 8.76253946j,
            3396.69851 + 1697.81748j,
            33.2706889 + 23.807742j,
        ],
        [
            -337.39782 + 187.765058j,
            33.2706889 + 23.807742j,
            3250.6316 + 221.971273j,
        ],
    ]
)


EXPECTED_REL_NEG1 = {
    "solver_output/q_R_output": np.array(
        [
            1.99387212,
            1.95826379,
            1.92983443,
            1.908841,
            1.89620117,
            1.89321553,
            1.89276882,
            1.89850559,
            1.91249229,
            1.93372322,
        ]
    ),
    "solver_output/q_Z_output": np.array(
        [
            -0.0780402519,
            -0.082802406,
            -0.088868817,
            -0.0964463656,
            -0.10591477,
            -0.111842019,
            -0.117717259,
            -0.132168381,
            -0.149264436,
            -0.168908009,
        ]
    ),
    "solver_output/q_zeta_output": np.array(
        [
            3.64302277e-17,
            8.77926377e-05,
            0.000324713594,
            0.000679998745,
            0.00111900011,
            0.00136743852,
            0.00158794487,
            0.00202335248,
            0.00237357384,
            0.00260650258,
        ]
    ),
    "solver_output/K_R_output": np.array(
        [
            -1146.40007,
            -932.547927,
            -723.841047,
            -496.097921,
            -237.308118,
            -91.9733148,
            37.1455266,
            294.311134,
            517.91264,
            719.289707,
        ]
    ),
    "solver_output/K_Z_output": np.array(
        [
            -120.491503,
            -156.42311,
            -196.094511,
            -244.943114,
            -307.418819,
            -345.725086,
            -381.753908,
            -459.570466,
            -534.724451,
            -609.163098,
        ]
    ),
    "solver_output/K_zeta_output": np.array(8.327130896577119e-14),
    "solver_output/tau": np.array(
        [
            0,
            22.7943506,
            45.5887013,
            68.3830519,
            91.1774025,
            103.237127,
            113.971753,
            136.766104,
            159.560454,
            182.354805,
        ]
    ),
    "analysis/B_magnitude": np.array(
        [
            0.756057183,
            0.769299733,
            0.780257904,
            0.78858547,
            0.793711851,
            0.794946712,
            0.795150087,
            0.792875709,
            0.78732962,
            0.779059451,
        ]
    ),
    "analysis/normalised_omega_ce": np.array(
        [
            0.384798594,
            0.386853892,
            0.389379275,
            0.391737052,
            0.393428032,
            0.393932766,
            0.394134006,
            0.39384941,
            0.39288215,
            0.39173332,
        ]
    ),
    "analysis/normalised_omega_pe": np.array(
        [
            0.000616658555,
            0.510602594,
            0.651547874,
            0.722588217,
            0.753847882,
            0.757673462,
            0.75409543,
            0.723292481,
            0.652639834,
            0.512128262,
        ]
    ),
}


def simple(path):
    """Built-in synthetic diagnostic: the analytical circular flux surfaces"""
    kwargs_dict = synthetic_parameters()
    kwargs_dict["find_B_method"] = "analytical"
    return kwargs_dict


def ne_dat_file(path):
    """Density fit using TORBEAM file"""
    kwargs_dict = synthetic_parameters()

    ne_filename = path / "ne.dat"
    rho = np.linspace(0, 1)
    density_fit = kwargs_dict["density_fit_method"]
    density = density_fit(rho**2)

    with open(ne_filename, "w") as f:
        f.write(f"{len(rho)}\n")
        np.savetxt(f, np.column_stack((rho, density)), fmt="%.7e")

    kwargs_dict["find_B_method"] = "analytical"
    kwargs_dict["density_fit_parameters"] = None
    kwargs_dict["density_fit_method"] = "smoothing-spline-file"
    kwargs_dict["ne_data_path"] = path
    return kwargs_dict


def synthetic_args():
    kwargs_dict = synthetic_parameters()
    field = CircularFluxSurfaceField(
        R_axis=kwargs_dict["R_axis"],
        minor_radius_a=kwargs_dict["minor_radius_a"],
        B_T_axis=kwargs_dict["B_T_axis"],
        B_p_a=kwargs_dict["B_p_a"],
        R_points=256,
        Z_points=256,
        grid_buffer_factor=1.1,
    )
    x_meshgrid, z_meshgrid = np.meshgrid(field.R_coord, field.Z_coord, indexing="ij")
    B_t = field.B_T(x_meshgrid, 0, z_meshgrid)
    B_r = field.B_R(x_meshgrid, 0, z_meshgrid)
    B_z = field.B_Z(x_meshgrid, 0, z_meshgrid)
    psi = field.polflux(x_meshgrid, 0, z_meshgrid)
    return kwargs_dict, B_r, B_t, B_z, psi, field


def torbeam_file(path):
    """Geometry using TORBEAM file"""
    kwargs_dict, B_R, B_t, B_Z, psi, field = synthetic_args()
    x_meshgrid, z_meshgrid = np.meshgrid(field.R_coord, field.Z_coord, indexing="ij")
    B_t = field.B_T(x_meshgrid, 0, z_meshgrid)
    B_r = field.B_R(x_meshgrid, 0, z_meshgrid)
    B_z = field.B_Z(x_meshgrid, 0, z_meshgrid)
    psi = field.polflux(x_meshgrid, 0, z_meshgrid)

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


def unnormalised_flux_scale(kwargs_dict: dict) -> float:
    """The unnormalised poloidal flux per unit normalised flux of
    CircularFluxSurfaceField, B_p_a a R_axis / 2, so that
    B_p = |grad(Psi)| / R"""
    return (
        kwargs_dict["B_p_a"] * kwargs_dict["minor_radius_a"] * kwargs_dict["R_axis"] / 2
    )


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
    scale = unnormalised_flux_scale(kwargs_dict)
    psi_axis = field.polflux(R_axis, 0, 0.0) * scale
    psi_boundary = field.polflux(R_axis + minor_radius, 0, 0.0) * scale

    R_midplane = np.linspace(R_axis, R_axis + minor_radius)
    rBphi = R_midplane * field.B_T(R_midplane, 0, 0.0)

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
    scale = unnormalised_flux_scale(kwargs_dict)
    psi_axis = field.polflux(R_axis, 0, 0.0) * scale
    psi_boundary = field.polflux(R_axis + minor_radius, 0, 0.0) * scale

    R_midplane = np.linspace(R_axis, R_axis + minor_radius)
    rBphi = R_midplane * field.B_T(R_midplane, 0, 0.0)
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


def EFITpp_file(path: pathlib.Path):
    """find_B_method = "EFITpp": a synthetic efitOut.nc with EFIT++'s layout"""
    import h5py

    kwargs_dict, B_R, B_t, B_Z, psi, field = synthetic_args()

    R_axis = kwargs_dict["R_axis"]
    minor_radius = kwargs_dict["minor_radius_a"]

    # A physically consistent file: the unnormalised flux is psi_unnorm =
    # (psi_boundary - psi_axis) psi_norm, with psi_boundary - psi_axis =
    # -unnormalised_flux_scale as in UDA_saved, so from_EFITpp's linear fit
    # gives dpsi_norm/dpsi = 1/slope
    slope = -unnormalised_flux_scale(kwargs_dict)
    R_midplane = np.linspace(R_axis, R_axis + minor_radius)
    psi_norm_1D = np.linspace(0, 1, len(R_midplane))
    rBphi = R_midplane * field.B_T(R_midplane, 0, 0.0)
    time = [100, 200, 300]

    # Written with h5py: h5netcdf's legacy API cannot write (or read)
    # variables two groups deep. EFIT++ output is netCDF4, i.e. HDF5
    with h5py.File(path / "efitOut.nc", "w") as efitpp:
        efitpp["time"] = time
        profiles2D = efitpp.create_group("output/profiles2D")
        profiles2D["poloidalFlux"] = add_timeslices(psi * slope)
        profiles2D["r"] = add_timeslices(field.R_coord)
        profiles2D["z"] = add_timeslices(field.Z_coord)
        flux = efitpp.create_group("output/fluxFunctionProfiles")
        flux["normalizedPoloidalFlux"] = psi_norm_1D
        flux["poloidalFlux"] = add_timeslices(psi_norm_1D * slope)
        flux["rBphi"] = add_timeslices(rBphi)

    kwargs_dict["find_B_method"] = "EFITpp"
    kwargs_dict["magnetic_data_path"] = path
    kwargs_dict["equil_time"] = time[1]
    kwargs_dict["delta_R"] = -0.0001
    kwargs_dict["delta_Z"] = 0.0001
    return kwargs_dict


def assert_is_symmetric(array: FloatArray) -> None:
    assert_allclose(array, array.T)


def check_Psi(
    result: xr.DataArray,
    expected_start: FloatArray,
    cutoff_index: int,
    expected_cutoff: Optional[FloatArray],
    expected_final: Optional[FloatArray],
):
    """Psi at the start, the cut-off and the end. ``expected_cutoff=None``
    or ``expected_final=None`` skips that point, for cases that check it
    separately"""
    assert_is_symmetric(result[0, ...])

    # Mixed zeta elements: comparitively large atol because these are
    # expected to be zero
    assert np.isclose(result[0, 0, 1], expected_start[0, 1], rtol=1e-2, atol=0.4)
    assert np.isclose(result[0, 1, 2], expected_start[1, 2], rtol=1e-2, atol=0.4)

    # Diagonal elements
    assert_allclose(
        result[0, ...].data.diagonal(), expected_start.diagonal(), rtol=1e-2, atol=0.1
    )
    # RZ element
    assert np.isclose(result[0, 0, 2], expected_start[0, 2], rtol=1e-2, atol=0.1)

    if expected_cutoff is not None:
        check_Psi_cutoff(result[cutoff_index, ...], expected_cutoff)
    if expected_final is not None:
        check_Psi_final(result[-1, ...], expected_final)


def check_Psi_cutoff(result: xr.DataArray, expected: FloatArray) -> None:
    assert_allclose(result, expected, rtol=1e-2, atol=0.1)


def check_Psi_final(result: xr.DataArray, expected: FloatArray) -> None:
    assert_allclose(result, expected, rtol=1.8e-2, atol=0.1)


def assert_ray_matches_golden(output, expected: dict, rtol=1e-2, atol=1e-2):
    """Compare the ray with the golden values at matched tau.

    The output grid is spaced tau_leave / (len_tau - 1), and tau_leave comes
    from the ray solver's exit event, which at the default tolerance differs
    between v3 and v4 by about 0.1%. Comparing the arrays index by index
    therefore compares slightly different points, which near a zero crossing
    of K_R (dK_R/dtau ~ 7 m^-1 here) gives differences of ~2 on values of
    ~10. So tau is compared on its own, and every other array is interpolated
    onto the golden tau first. On the same field, v3 and v4 at tight tolerance
    agree to 3e-4 in K_R; at matched tau the default-tolerance run is within
    0.1% of the golden values.
    """
    tau_key = "solver_output/tau"
    tau_expected = expected[tau_key]
    tau = np.asarray(output[tau_key])
    assert_allclose(tau, tau_expected, rtol=rtol, atol=atol, err_msg=tau_key)

    for key, value in expected.items():
        if key == tau_key:
            continue
        result = np.asarray(output[key])
        if np.ndim(value) == 0 or np.ndim(result) == 0:
            assert_allclose(result, value, rtol=rtol, atol=atol, err_msg=key)
        else:
            matched = np.interp(tau_expected, tau, result)
            assert_allclose(matched, value, rtol=rtol, atol=atol, err_msg=key)


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

    output = beam_me_up(**kwargs_dict)

    for key, value in EXPECTED_1.items():
        assert_allclose(output[key], value, rtol=1e-2, atol=1e-2, err_msg=key)

    K_magnitude = np.hypot(
        output["solver_output/K_R_output"], output["solver_output/K_Z_output"]
    )
    assert K_magnitude.argmin("tau") == CUTOFF_INDEX_1

    check_Psi(
        output["solver_output/Psi_3D_output_labframe_cylindrical"],
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

    output = beam_me_up(**kwargs_dict)

    assert_ray_matches_golden(output, EXPECTED_NEG1)

    K_magnitude = np.hypot(
        output["solver_output/K_R_output"], output["solver_output/K_Z_output"]
    )
    assert K_magnitude.argmin("tau") == CUTOFF_INDEX_NEG1

    # Psi at the cut-off: test_X_mode_cutoff_Psi, at rtol = 1e-6
    check_Psi(
        output["solver_output/Psi_3D_output_labframe_cylindrical"],
        PSI_START_EXPECTED_NEG1,
        CUTOFF_INDEX_NEG1,
        None,
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

    output = beam_me_up(**kwargs_dict)

    # At matched tau: tau_leave differs from v3 by 0.25%, and K_R crosses zero
    # near the end, so index-by-index comparison compares different points
    assert_ray_matches_golden(output, EXPECTED_REL_1)

    K_magnitude = np.hypot(
        output["solver_output/K_R_output"], output["solver_output/K_Z_output"]
    )
    assert K_magnitude.argmin("tau") == CUTOFF_INDEX_REL_1

    # Psi at the final point: test_relativistic_O_mode_final_Psi, at rtol = 1e-6
    check_Psi(
        output["solver_output/Psi_3D_output_labframe_cylindrical"],
        PSI_START_EXPECTED_REL_1,
        CUTOFF_INDEX_REL_1,
        PSI_CUTOFF_EXPECTED_REL_1,
        None,
    )


@pytest.mark.parametrize(
    "generator_relfinal",
    [
        pytest.param(simple, id="simple"),
        pytest.param(ne_dat_file, id="density-fit-file"),
        pytest.param(torbeam_file, id="torbeam-file"),
        pytest.param(npz_file, id="test-file"),
    ],
)
def test_relativistic_O_mode_final_Psi(tmp_path, generator_relfinal):
    """Psi at the final point for test_relativistic_O_mode, at rtol = 1e-6
    against v3 at rtol = 1e-6. The final point sits at tau_leave, which at
    the default rtol = 1e-3 differs from v3's by 2.5e-3 (Psi 2.95% apart);
    at rtol = 1e-6 in both codes, 2.4e-7 (Psi 0.26% apart)"""
    kwargs_dict = generator_relfinal(tmp_path)
    kwargs_dict["output_filename_suffix"] = "_Bpa1.12"
    kwargs_dict["figure_flag"] = False
    kwargs_dict["len_tau"] = 10
    kwargs_dict["output_path"] = tmp_path
    kwargs_dict["relativistic_flag"] = True
    kwargs_dict["temperature_fit_method"] = QuadraticFit(1.0, 10.0)
    kwargs_dict["rtol"] = 1e-6

    output = beam_me_up(**kwargs_dict)

    assert_allclose(
        output["solver_output/tau"][-1],
        TAU_LEAVE_EXPECTED_REL_1_RTOL_1E_6,
        rtol=1e-2,
        atol=1e-2,
    )
    check_Psi_final(
        output["solver_output/Psi_3D_output_labframe_cylindrical"][-1],
        PSI_FINAL_EXPECTED_REL_1_RTOL_1E_6,
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

    output = beam_me_up(**kwargs_dict)

    assert_ray_matches_golden(output, EXPECTED_REL_NEG1)

    K_magnitude = np.hypot(
        output["solver_output/K_R_output"], output["solver_output/K_Z_output"]
    )
    assert K_magnitude.argmin("tau") == CUTOFF_INDEX_REL_NEG1

    # The cut-off tau on its own, then Psi there against v3's Psi at the same tau
    tau_cutoff = float(output["solver_output/tau"][CUTOFF_INDEX_REL_NEG1])
    assert_allclose(tau_cutoff, TAU_CUTOFF_EXPECTED_REL_NEG1, rtol=2e-3)
    check_Psi(
        output["solver_output/Psi_3D_output_labframe_cylindrical"],
        PSI_START_EXPECTED_REL_NEG1,
        CUTOFF_INDEX_REL_NEG1,
        psi_cutoff_expected_rel_neg1(tau_cutoff),
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

    output = beam_me_up(**kwargs_dict)

    for key, value in EXPECTED_1.items():
        assert_allclose(output[key], value, rtol=1e-2, atol=1e-2, err_msg=key)

    K_magnitude = np.hypot(
        output["solver_output/K_R_output"], output["solver_output/K_Z_output"]
    )
    assert K_magnitude.argmin("tau") == CUTOFF_INDEX_1

    check_Psi(
        output["solver_output/Psi_3D_output_labframe_cylindrical"],
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

    output = beam_me_up(**kwargs_dict)

    assert_ray_matches_golden(output, EXPECTED_NEG1)

    K_magnitude = np.hypot(
        output["solver_output/K_R_output"], output["solver_output/K_Z_output"]
    )
    assert K_magnitude.argmin("tau") == CUTOFF_INDEX_NEG1

    # Psi at the cut-off: test_X_mode_cutoff_Psi, at rtol = 1e-6
    check_Psi(
        output["solver_output/Psi_3D_output_labframe_cylindrical"],
        PSI_START_EXPECTED_NEG1,
        CUTOFF_INDEX_NEG1,
        None,
        PSI_FINAL_EXPECTED_NEG1,
    )


@pytest.mark.parametrize(
    "temperature",
    [
        pytest.param(None, id="nonrelativistic"),
        pytest.param(0.0, id="null-relativistic"),
    ],
)
@pytest.mark.parametrize(
    "generator_cutoff",
    [
        pytest.param(simple, id="simple"),
        pytest.param(ne_dat_file, id="density-fit-file"),
        pytest.param(torbeam_file, id="torbeam-file"),
        pytest.param(npz_file, id="test-file"),
    ],
)
def test_X_mode_cutoff_Psi(tmp_path, generator_cutoff, temperature):
    """Psi at the cut-off for test_integrated_X_mode and
    test_null_relativistic_X_mode, at rtol = 1e-6 against v3 at rtol = 1e-6.
    It is steep there, and at the default rtol = 1e-3 both codes are further
    from the converged value (v3 0.35%, v4 0.9%) than from each other at
    rtol = 1e-6 (0.2%)"""
    kwargs_dict = generator_cutoff(tmp_path)
    kwargs_dict["output_filename_suffix"] = "_Bpa0.12"
    kwargs_dict["figure_flag"] = False
    kwargs_dict["len_tau"] = 10
    kwargs_dict["output_path"] = tmp_path
    kwargs_dict["mode_flag"] = -1
    kwargs_dict["rtol"] = 1e-6
    if temperature is not None:
        kwargs_dict["relativistic_flag"] = True
        kwargs_dict["temperature_fit_method"] = QuadraticFit(1.0, temperature)

    output = beam_me_up(**kwargs_dict)

    K_magnitude = np.hypot(
        output["solver_output/K_R_output"], output["solver_output/K_Z_output"]
    )
    assert K_magnitude.argmin("tau") == CUTOFF_INDEX_NEG1
    assert_allclose(
        output["solver_output/tau"][CUTOFF_INDEX_NEG1],
        TAU_CUTOFF_EXPECTED_NEG1_RTOL_1E_6,
        rtol=1e-2,
        atol=1e-2,
    )
    check_Psi_cutoff(
        output["solver_output/Psi_3D_output_labframe_cylindrical"][CUTOFF_INDEX_NEG1],
        PSI_CUTOFF_EXPECTED_NEG1_RTOL_1E_6,
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
        pytest.param(EFITpp_file, id="EFITpp-file"),
    ],
)
def test_create_magnetic_geometry(tmp_path, generator):
    kwargs_dict = generator(tmp_path)

    field = cast(MagneticField_Cylindrical, make_field(kwargs_dict))
    field_golden = cast(MagneticField_Cylindrical, make_field(simple(tmp_path)))

    R_axis = kwargs_dict["R_axis"]
    minor_radius = kwargs_dict["minor_radius_a"]
    R = np.linspace(R_axis + 0.1, R_axis + minor_radius, 10)
    Z = np.linspace(0.1, kwargs_dict["minor_radius_a"], 10)

    assert_allclose(
        field.B_R(R_axis, 0, Z), field_golden.B_R(R_axis, 0, Z), rtol=1e-3, atol=1e-3
    )
    assert_allclose(
        field.B_T(R, 0, 0.0), field_golden.B_T(R, 0, 0.0), rtol=1e-3, atol=1e-3
    )
    assert_allclose(
        field.B_Z(R, 0, 0.0), field_golden.B_Z(R, 0, 0.0), rtol=1e-3, atol=1e-3
    )
    assert_allclose(
        field.polflux(R, 0, 0.0),
        field_golden.polflux(R, 0, 0.0),
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.parametrize(
    "generator",
    [
        pytest.param(UDA_saved, id="UDA-saved-file"),
        pytest.param(EFITpp_file, id="EFITpp-file"),
    ],
)
def test_EFIT_geometry(tmp_path, generator):
    kwargs_dict = generator(tmp_path)
    field = cast(MagneticField_Cylindrical, make_field(kwargs_dict))
    field_golden = cast(MagneticField_Cylindrical, make_field(simple(tmp_path)))

    R_axis = kwargs_dict["R_axis"]
    minor_radius_a = kwargs_dict["minor_radius_a"]
    scale = unnormalised_flux_scale(kwargs_dict)

    # Include buffer for gradient near edge
    width = minor_radius_a + 0.05
    # Different grid sizes to capture transpose errors
    R = np.linspace(R_axis - width, R_axis + width, 99)
    Z = np.linspace(-width, width, 101)
    R_grid, Z_grid = np.meshgrid(R, Z, indexing="ij")
    B_R = field.B_R(R_grid, 0, Z_grid)
    B_Z = field.B_Z(R_grid, 0, Z_grid)

    psi = field.polflux(R_grid, 0, Z_grid)
    grad_psi = np.gradient(psi, R, Z)
    calculated_B_R = grad_psi[1] * scale / R_grid
    calculated_B_Z = -grad_psi[0] * scale / R_grid

    mask = (psi > 0.1) & (psi <= 1)

    assert_allclose(B_R[mask], calculated_B_R[mask], 2e-3, 2e-3)
    assert_allclose(B_Z[mask], calculated_B_Z[mask], 2e-3, 2e-3)

    # Inside the plasma only, as for B_R and B_Z: outside, the readers' spline
    # of rBphi(psi) is extrapolated (2e-7 relative at psi ~ 2.4)
    assert_allclose(
        field.B_T(R_grid, 0, Z_grid)[mask], field_golden.B_T(R_grid, 0, Z_grid)[mask]
    )

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
    """The same golden values as v3's test_launch_golden_answer, called the v4
    way: a ProfileFit object goes through make_fit (as in beam_me_up, and as
    v3's make_density_fit did), and find_plasma_entry_parameters is given the
    Hamiltonians for both modes, as beam_me_up does"""
    args = generator(tmp_path)
    field = make_field(args)
    ne_data_path = args.get("ne_data_path", None)
    if ne_data_path:
        ne_filename = ne_data_path / "ne.dat"
        parameters = [ne_filename, 5, 0]
    else:
        parameters = None
        ne_filename = None

    density_fit = make_fit(
        args["density_fit_method"],
        args["poloidal_flux_zero_density"],
        parameters,
        ne_filename,
    )

    # v3's steps: delta_R = -1e-3 (into the plasma), delta_Z = 1e-3, delta_K = 0.1
    hamiltonian_pos1, hamiltonian_neg1 = initialise_hamiltonians(
        launch_angular_freq=freq_GHz_to_angular_frequency(args["launch_freq_GHz"]),
        deltas=np.array([-1e-3, 1e-3, 0.1, 0.1, 0.1]),
        field=field,
        density_fit=density_fit,
        temperature_fit=None,
    )

    q_initial = find_plasma_entry_position(
        args["poloidal_launch_angle_Torbeam"],
        args["toroidal_launch_angle_Torbeam"],
        args["launch_position"],
        "vacuum_analytic",
        field,
        args["poloidal_flux_enter"],
    )

    (
        K_launch,
        K_initial,
        Psi_3D_launch_labframe,
        Psi_3D_entry_labframe,
        Psi_3D_initial_labframe,
        distance_from_launch_to_entry,
        e_hat_initial,
        mode_flag_initial,
        mode_index,
    ) = find_plasma_entry_parameters(
        ray_tracing_flag=False,
        launch_flag="vacuum_analytic",
        Psi_BC_flag=args["Psi_BC_flag"],
        mode_flag_launch=args["mode_flag"],
        poloidal_launch_angle_deg_Torbeam=args["poloidal_launch_angle_Torbeam"],
        toroidal_launch_angle_deg_Torbeam=args["toroidal_launch_angle_Torbeam"],
        q_launch=args["launch_position"],
        q_initial=q_initial,
        launch_beam_width=args["launch_beam_width"],
        launch_beam_curvature=args["launch_beam_curvature"],
        field=field,
        hamiltonian_pos1=hamiltonian_pos1,
        hamiltonian_neg1=hamiltonian_neg1,
    )

    expected_q_initial = np.array([1.99382564, -0.0, -0.07804514])
    expected_K_launch = np.array([-1146.40007, -0.0, -120.49150267])
    expected_K_initial = np.array([-1146.4000699962507, -0.0, -120.4915026654739])
    # Note that R-zeta, Z-zeta terms are not actually zero, but should be
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
    expected_Psi_3D_lab_entry_cartesian = np.array(
        [
            [5.38751589 + 11.95597601j, 0.0 + 0.0j, -51.25878964 - 113.75351313j],
            [0.0 + 0.0j, 493.082322 + 1094.24835789j, 0.0 + 0.0j],
            [-51.25878964 - 113.75351313j, 0.0 + 0.0j, 487.69480611 + 1082.29238188j],
        ]
    )
    expected_Psi_3D_lab_initial = np.array(
        [
            [-4.955445e03 + 1.195869e01j, 0.0 + 0.0j, 7.326383e02 - 1.137689e02j],
            [0.0 + 0.0j, 4.245883e03 + 4.350387e03j, 0.0 + 0.0j],
            [7.326383e02 - 1.137689e02j, 0.0 + 0.0j, 3.637885e02 + 1.082339e03j],
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
    assert_allclose(q_initial, expected_q_initial, tol, tol)
    assert_allclose(K_launch, expected_K_launch, tol, tol)
    assert_allclose(K_initial, expected_K_initial, tol, tol)
    assert_allclose(Psi_3D_launch_labframe, expected_Psi_3D, tol, tol)  # type: ignore
    assert_allclose(Psi_3D_launch_labframe, expected_Psi_3D_lab_launch, tol, tol)  # type: ignore
    assert_allclose(Psi_3D_entry_labframe, expected_Psi_3D_lab_entry, tol, tol)  # type: ignore
    # assert_allclose(Psi_3D_lab_entry_cartesian, expected_Psi_3D_lab_entry_cartesian, tol, tol) # TODO fix this
    assert_allclose(
        Psi_3D_initial_labframe, expected_Psi_3D_lab_initial, tol, atol=0.4
    )  # Larger atol due to some small values
    assert_allclose(distance_from_launch_to_entry, expected_distance_from_launch_to_entry, tol, tol)  # type: ignore


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
    return np.array(start_point), -phi_p, phi_t, expected_entry_cartesian


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
        pytest.param(
            np.array([2.5, 0, 0]), 0, 0, [2.0, 0.0, 0], id="outboard-midplane"
        ),
        pytest.param(
            np.array([0.5, 0, 0]), 0, np.pi, [1.0, 0, 0], id="inboard-midplane"
        ),
        pytest.param(np.array([1.5, 0, 1]), np.pi / 2, 0, [1.5, 0, 0.5], id="top"),
        pytest.param(
            np.array([1.5, 0, -1]), -np.pi / 2, 0, [1.5, 0, -0.5], id="bottom"
        ),
        pytest.param(
            np.array([2, 0, 0.5]),
            np.pi / 4,
            0,
            [1.5 + np.sqrt(0.5) / 2, 0, np.sqrt(0.5) / 2],
            id="top-right",
        ),
        pytest.param(
            np.array([2, 0, -0.5]),
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
def test_find_plasma_entry_position(
    tmp_path,
    generator,
    launch_position,
    poloidal_launch_angle,
    toroidal_launch_angle,
    expected_entry,
):
    args = generator(tmp_path)
    field = make_field(args)
    poloidal_flux_enter = args["poloidal_flux_enter"]

    # The launch angles above are in radians, but v4 takes degrees
    entry_position = find_plasma_entry_position(
        np.rad2deg(poloidal_launch_angle),
        np.rad2deg(toroidal_launch_angle),
        launch_position,
        "vacuum_analytic",
        field,
        poloidal_flux_enter,
    )

    assert_allclose(entry_position, expected_entry, 1e-6, 1e-6)


def test_find_plasma_entry_position_bug(tmp_path):
    args = simple(tmp_path)
    field = make_field(args)
    R, zeta, Z = find_plasma_entry_position(
        poloidal_launch_angle_deg_Torbeam=-1.0,
        toroidal_launch_angle_deg_Torbeam=0.0,
        q_launch=np.array([2.3, 0, -0.1]),
        launch_flag="vacuum_analytic",
        field=field,
        poloidal_flux_enter=1.0,
    )

    assert np.isclose(zeta, 0.0)


@pytest.mark.parametrize(
    ("q_initial", "expected_signs"),
    [
        # Outboard, below the midplane: the plasma is at smaller R and larger Z
        pytest.param(np.array([1.99, 0, -0.08]), (-1, 1), id="outboard-below"),
        # Outboard, above the midplane: smaller R and smaller Z
        pytest.param(np.array([1.99, 0, 0.08]), (-1, -1), id="outboard-above"),
        # Inboard, above the midplane: larger R and smaller Z
        pytest.param(np.array([1.01, 0, 0.08]), (1, -1), id="inboard-above"),
    ],
)
def test_auto_delta_signs_point_into_plasma(tmp_path, q_initial, expected_signs):
    """With `auto_delta_sign`, the finite-difference steps used for the
    one-sided Hamiltonian derivatives must point into the plasma, or the
    boundary conditions use the vacuum side of the density discontinuity"""
    field = make_field(simple(tmp_path))
    deltas = np.array([1e-4, 1e-4])

    flipped = find_auto_delta_signs(True, q_initial, deltas, field)
    assert_allclose(flipped, np.array(expected_signs) * 1e-4)

    unchanged = find_auto_delta_signs(False, q_initial, deltas, field)
    assert_allclose(unchanged, deltas)


def test_beam_me_up_uses_flipped_deltas(tmp_path):
    """The flipped steps must reach the Hamiltonian, not just the saved
    inputs: the initial Psi then matches v3's (which used delta_R < 0)"""
    kwargs_dict = simple(tmp_path)
    kwargs_dict["figure_flag"] = False
    kwargs_dict["len_tau"] = 10
    kwargs_dict["output_path"] = tmp_path
    kwargs_dict["delta_R"] = 1e-4  # pointing out of the plasma at entry

    output = beam_me_up(**kwargs_dict)

    assert float(output["inputs/delta_R"]) < 0
    assert_allclose(
        output["solver_output/Psi_3D_output_labframe_cylindrical"][0],
        PSI_START_EXPECTED_1,
        rtol=1e-3,
        atol=0.1,
    )


# @pytest.mark.parametrize(
#     "generator",
#     [pytest.param(simple, id="simple")],
# )
# def test_quick_run(tmp_path, generator):
#     """Golden answer test to check basic functionality using circular
#     flux surfaces with quick_run=True"""

#     kwargs_dict = generator(tmp_path)
#     kwargs_dict["output_filename_suffix"] = "_Bpa0.10"
#     kwargs_dict["figure_flag"] = False
#     kwargs_dict["len_tau"] = 10
#     kwargs_dict["output_path"] = tmp_path
#     kwargs_dict["quick_run"] = True

#     q_R_cutoff_expected = 1.694
#     q_Z_cutoff_expected = -0.18097
#     K_norm_expected = 561.52
#     poloidal_flux_cutoff_expected = 0.53061
#     theta_m_cutoff_expected = 0.132552

#     cutoff = beam_me_up(**kwargs_dict)

#     assert np.isclose(cutoff.q_R, q_R_cutoff_expected)
#     assert np.isclose(cutoff.q_Z, q_Z_cutoff_expected)
#     assert np.isclose(cutoff.K_norm_min, K_norm_expected)
#     assert np.isclose(cutoff.poloidal_flux, poloidal_flux_cutoff_expected)
#     assert np.isclose(cutoff.theta_m, theta_m_cutoff_expected)
