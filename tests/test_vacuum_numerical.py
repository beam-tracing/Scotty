"""launch_flag = "vacuum_numerical" (v3's vacuumLaunch_flag = True with
vacuum_propagation_flag = False): the solvers start at the antenna and integrate
through the vacuum, across the plasma boundary, into the plasma, with no boundary
conditions.

The v3 values are from v3 (upstream/master feb9499) on the DBS_synthetic preset
with vacuum_propagation_flag = False, Psi_BC_flag = None, mode_flag = -1,
len_tau = 102, rtol = 1e-6, atol = 1e-8, at seven of its output points. (v3's own
analysis fails in this mode, as Psi_3D_lab_entry is None, so they are its solver
output.) At these tolerances v4 agrees with them to 6e-8 in q and K, 2e-8 in the
exit tau and 5e-5 in Psi (relative to max |Psi| at each point). At v3's default
rtol = 1e-3 both codes are much less accurate here, as the beam crosses the
density's kink at the plasma boundary (psi = 1): for mode_flag = 1, v3's and
v4's exit tau differ from the converged value by 10% and 4%.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from scotty.beam_me_up import beam_me_up
from scotty.init_bruv import get_parameters_for_Scotty

V3_TAU_LAST = 528.7420543242285
V3_TAU = np.array(
    [
        0.0,
        105.74841086484571,
        211.49682172969142,
        317.24523259453713,
        422.99364345938284,
        523.4546337809863,
        528.7420543242285,
    ]
)
V3_q_R = np.array(
    [
        2.587,
        2.4045279549361678,
        2.2220559098723376,
        2.039583864808506,
        1.9041221431286608,
        1.9516804906257477,
        1.958546132615667,
    ]
)
V3_q_Z = np.array(
    [
        -0.0157,
        -0.03487858475377839,
        -0.05405716950755679,
        -0.07323575426133516,
        -0.10117778844354552,
        -0.17133462914279352,
        -0.17646358926965133,
    ]
)
V3_K_R = np.array(
    [
        -1146.4000699962507,
        -1146.4000699962507,
        -1146.4000699962507,
        -1146.4000699962507,
        -310.31034662784646,
        838.7406626718878,
        887.1199043651714,
    ]
)
V3_K_Z = np.array(
    [
        -120.4915026654739,
        -120.4915026654739,
        -120.4915026654739,
        -120.4915026654739,
        -286.987882400557,
        -635.020690508655,
        -653.4578096236021,
    ]
)
V3_PSI = np.array(
    [
        [
            [
                -3.1486979026080815 + 13.657749541371441j,
                0.0 + 0.0j,
                29.957859401659583 - 129.94480676109941j,
            ],
            [0.0 + 0.0j, 1037.0812104565562 + 8365.71125j, 0.0 + 0.0j],
            [
                29.957859401659583 - 129.94480676109941j,
                0.0 + 0.0j,
                -285.0299926157482 + 1236.3422504586285j,
            ],
        ],
        [
            [
                -0.3020114048185155 + 14.37731556872815j,
                -4.77809966133765e-07 - 1.3933411112598394e-06j,
                2.873446574775127 - 136.79102019424818j,
            ],
            [
                -4.77809966133765e-07 - 1.3933411112598394e-06j,
                2596.73739533436 + 7607.961973929631j,
                1.8828435563307514e-08 - 5.9880966465162914e-09j,
            ],
            [
                2.873446574775127 - 136.79102019424818j,
                1.8828435563307514e-08 - 5.9880966465162914e-09j,
                -27.339017952148236 + 1301.4796201930005j,
            ],
        ],
        [
            [
                2.6186839395721173 + 13.889957226413063j,
                -1.2961374741879943e-06 - 2.171722170220888e-06j,
                -24.91511339150851 - 132.1541153056567j,
            ],
            [
                -1.2961374741879943e-06 - 2.171722170220888e-06j,
                3730.7442264445285 + 6276.8498279968535j,
                3.321019857730728e-08 - 2.9281290573865628e-08j,
            ],
            [
                -24.91511339150851 - 132.1541153056567j,
                3.321019857730728e-08 - 2.9281290573865628e-08j,
                237.05146922509385 + 1257.3624171433737j,
            ],
        ],
        [
            [
                4.942492243661963 + 12.416213681415897j,
                -2.55204436826937e-06 - 2.8438582300189074e-06j,
                -47.024672518368185 - 118.13238210649558j,
            ],
            [
                -2.55204436826937e-06 - 2.8438582300189074e-06j,
                4219.92074742185 + 4727.1912558865315j,
                4.5880141815523275e-08 - 9.186427106767278e-08j,
            ],
            [
                -47.024672518368185 - 118.13238210649558j,
                4.5880141815523275e-08 - 9.186427106767278e-08j,
                447.40987268022025 + 1123.954537206682j,
            ],
        ],
        [
            [
                -33476.74457772172 + 897.243308780821j,
                180.28302230738325 + 112.921406510234j,
                5565.279880331944 - 959.1403200978817j,
            ],
            [
                180.28302230738325 + 112.921406510234j,
                4036.607261613809 + 2927.287763500081j,
                -8.387122806087612 + 13.067052449425852j,
            ],
            [
                5565.279880331944 - 959.1403200978817j,
                -8.387122806087612 + 13.067052449425852j,
                1333.3847437542024 + 1031.4500403802765j,
            ],
        ],
        [
            [
                7057.085858443572 + 125.13424515950717j,
                11.92787088435714 + 11.454250835865668j,
                -201.75023403091836 + 165.21610640708485j,
            ],
            [
                11.92787088435714 + 11.454250835865668j,
                3438.0951900340137 + 1753.921326742553j,
                28.95209409781719 + 21.857076449754764j,
            ],
            [
                -201.75023403091836 + 165.21610640708485j,
                28.95209409781719 + 21.857076449754764j,
                3329.983203899274 + 218.1640209443278j,
            ],
        ],
        [
            [
                6761.277784803306 + 112.52096432576917j,
                14.976915455760782 + 12.577353729539697j,
                -182.09739532978935 + 152.7456009650972j,
            ],
            [
                14.976915455760782 + 12.577353729539697j,
                3417.6901461178177 + 1726.014721281315j,
                26.965947208438067 + 20.423382822412343j,
            ],
            [
                -182.09739532978935 + 152.7456009650972j,
                26.965947208438067 + 20.423382822412343j,
                3347.9451607117 + 207.35792600853222j,
            ],
        ],
    ]
)


def numerical(tmp_path, **kwargs):
    parameters = get_parameters_for_Scotty("DBS_synthetic")
    parameters.update(
        launch_flag="vacuum_numerical",
        Psi_BC_flag=None,
        console_log_level="WARNING",
        output_path=tmp_path,
    )
    parameters.update(kwargs)
    return beam_me_up(**parameters)


@pytest.fixture(scope="module", params=[1, -1], ids=["mode_flag=1", "mode_flag=-1"])
def run(request, tmp_path_factory):
    return numerical(
        tmp_path_factory.mktemp("vacuum_numerical"), mode_flag=request.param, len_tau=22
    )


def test_starts_at_the_antenna(run):
    """q, K and Psi where the solvers start are the launched ones"""
    so = run["solver"]
    q_launch = so["q_launch_cylindrical"].values
    assert_allclose(so["q_initial_cylindrical"].values, q_launch, rtol=0, atol=0)
    assert_allclose(so["q_output_cylindrical"].values[0], q_launch, atol=1e-12)
    assert_allclose(
        so["K_initial_cylindrical"].values, so["K_launch_cylindrical"].values, atol=0
    )
    assert_allclose(
        so["Psi_3D_initial_labframe_cylindrical"].values,
        so["Psi_3D_launch_labframe_cylindrical"].values,
        atol=0,
    )
    # In vacuum, K is the launched K (Cartesian, as K_zeta varies with R)
    assert_allclose(
        so["K_output_cartesian"].values[0], so["K_launch_cartesian"].values, rtol=1e-12
    )


def test_entry_quantities_are_nan(run):
    so = run["solver"]
    assert np.isnan(float(so["distance_from_launch_to_entry"]))
    for frame in ("cartesian", "cylindrical"):
        assert np.isnan(so[f"Psi_3D_entry_labframe_{frame}"].values).all()


def test_reaches_the_plasma_and_is_analysed(run):
    """The ray crosses the boundary into the plasma (psi < 1) and the analysis
    completes. The launched beam has a waist, so filterfun_b is finite"""
    assert run["beam"]["polflux"].values[0] > 1
    assert run["beam"]["polflux"].values.min() < 0.9
    dbs = run["diagnostics/dbs"]
    assert np.isfinite(float(run["beam"]["beam_waist"]))
    assert np.isfinite(dbs["filterfun_b"].values).all()


def test_ray_tracing_only(tmp_path):
    dt = numerical(tmp_path, ray_tracing_flag=True, len_tau=12)
    so = dt["solver"]
    assert_allclose(
        so["q_initial_cylindrical"].values, so["q_launch_cylindrical"].values, atol=0
    )
    assert dt["beam"]["polflux"].values.min() < 0.9


def test_delta_signs_set_at_the_entry_point(tmp_path):
    """As in v3, the spatial steps point into the plasma at the plasma entry,
    not at the antenna, though the solvers start at the antenna"""
    dt = numerical(tmp_path, ray_tracing_flag=True, len_tau=12, delta_R=1e-5)
    assert float(dt["inputs"]["delta_R"]) == -1e-5


def test_matches_v3(tmp_path):
    """Against v3 with vacuum_propagation_flag = False, at tight tolerances (see
    the module docstring)"""
    dt = numerical(
        tmp_path,
        mode_flag=-1,
        len_tau=102,
        rtol=1e-6,
        atol=1e-8,
        # v3's default steps
        delta_R=-1e-4,
        delta_Z=1e-4,
    )
    so = dt["solver"]
    tau = so["tau"].values
    assert_allclose(tau[-1], V3_TAU_LAST, rtol=1e-6)

    for v3_values, name in (
        (V3_q_R, "q_R_output"),
        (V3_q_Z, "q_Z_output"),
        (V3_K_R, "K_R_output"),
        (V3_K_Z, "K_Z_output"),
    ):
        v4_values = np.interp(V3_TAU, tau, so[name].values)
        assert_allclose(
            v4_values, v3_values, rtol=0, atol=1e-6 * np.abs(v3_values).max()
        )

    Psi = so["Psi_3D_output_labframe_cylindrical"].values
    for k, tau_v3 in enumerate(V3_TAU):
        Psi_v4 = np.array(
            [
                [
                    np.interp(tau_v3, tau, Psi[:, i, j].real)
                    + 1j * np.interp(tau_v3, tau, Psi[:, i, j].imag)
                    for j in range(3)
                ]
                for i in range(3)
            ]
        )
        scale = np.abs(V3_PSI[k]).max()
        assert_allclose(Psi_v4, V3_PSI[k], rtol=0, atol=1e-3 * scale)
