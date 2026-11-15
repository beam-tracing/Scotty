"""Tests for the fixes made before 4.0.0rc, from the code audit of 14 Nov 2026.

Each test names the audit item it covers (B1, B2, ... in the audit's
numbering). Each one fails on master before the fix.
"""

import inspect
import logging

import numpy as np
import pytest
from numpy.testing import assert_allclose

from scotty.beam_me_up import beam_me_up
from scotty.compat import from_v3_kwargs
from scotty.fun_general import launch_direction_cartesian, ray_line
from scotty.init_bruv import get_parameters_for_Scotty
from scotty.profile_fit import (
    PolynomialFit,
    QuadraticFit,
    SmoothingSplineFit,
    StefanikovaFit,
    TanhFit,
)


def synthetic(tmp_path, **kwargs):
    parameters = get_parameters_for_Scotty("DBS_synthetic")
    parameters.update(output_path=tmp_path, len_tau=52)
    parameters.update(kwargs)
    return parameters


# ---------------------------------------------------------------- profiles


def test_B3_quadratic_is_zero_at_psi_0():
    fit = QuadraticFit(1.01, 4.0)
    assert fit(1.01) == 0.0
    psi = np.linspace(0.0, 1.01, 102)[:-1]
    assert np.all(fit(psi) > 0)
    assert_allclose(fit(psi), 4.0 * (1 - (psi / 1.01) ** 2), rtol=1e-14)


def test_B3_quadratic_unchanged_at_psi_0_one():
    """The DBS_synthetic preset (psi_0 = 1) gives the same profile as before"""
    psi = np.linspace(0.0, 1.0, 11)
    assert_allclose(QuadraticFit(1.0, 4.0)(psi), 4.0 - 4.0 * psi**2, rtol=1e-14)


def test_B5_stefanikova_core_follows_the_formula():
    """With the pedestal flat at zero, the profile is a_height exp(-(psi/w)^p)"""
    fit = StefanikovaFit(1.0, 2.0, 0.5, 2.0, 0.0, 0.0, 0.1, 0.0, 5.0)
    psi = np.array([0.0, 0.25, 0.5, 0.75])
    pedestal = fit._f_ped(psi)
    expected = pedestal + (2.0 - pedestal) * np.exp(-((psi / 0.5) ** 2))
    assert_allclose(fit(psi), expected, rtol=1e-14)
    # The audit's example: exp(-1), not the old exp(-2), at psi = 0.5
    assert_allclose(fit(0.5) - pedestal[2], (2.0 - pedestal[2]) * np.exp(-1.0))


def test_B4_spline_holds_its_end_value():
    """A file that ends at psi = 1 with a non-zero slope gives no negative
    density past its data"""
    psi = np.linspace(0, 1, 50)
    fit = SmoothingSplineFit(1.01, psi, 4.0 * (1 - psi**2), 5)
    outside = fit(np.array([1.003, 1.006, 1.0099]))
    assert np.all(outside >= 0)
    assert_allclose(outside, fit(1.0), atol=1e-12)


def test_B4_spline_goes_through_its_data_by_default():
    psi = np.linspace(0, 1, 40)
    data = 1 - psi**2 + 0.3 * np.exp(-(((psi - 0.5) / 0.05) ** 2))
    assert_allclose(SmoothingSplineFit(1.0, psi, data, 3)(psi), data, atol=1e-10)


def test_profiles_clipped_at_zero_with_one_warning(caplog):
    # A tanh profile with the "wrong" sign of ne_1 is negative everywhere inside
    fit = TanhFit(1.0, 2.0, 1.0)
    with caplog.at_level(logging.WARNING, logger="scotty.profile_fit"):
        assert np.all(fit(np.linspace(0, 1, 11)) == 0)
        fit(0.5)
    assert len([r for r in caplog.records if "clipping" in r.message]) == 1


# ---------------------------------------------------------------- outputs


def test_B1_curvature_at_the_antenna_is_the_launch_curvature(tmp_path):
    """At the antenna (`vacuum_numerical` starts there), the saved widths and
    curvatures are the launched ones: 0.04 m and -0.25 /m for DBS_synthetic.
    Before the fix the curvatures were |g|^2 (about 3e-6) times too small"""
    dt = beam_me_up(
        **synthetic(tmp_path, launch_flag="vacuum_numerical", Psi_BC_flag=None)
    )
    beam = dt["beam"].ds
    for name, value in (
        ("width_1", 0.04),
        ("width_2", 0.04),
        ("curvature_1", -0.25),
        ("curvature_2", -0.25),
    ):
        assert_allclose(beam[name].values[0], value, rtol=1e-6)
    # B11: width_1 is the larger width at each point, curvature_1 the smaller
    assert np.all(beam["width_1"].values >= beam["width_2"].values)
    assert np.all(beam["curvature_1"].values <= beam["curvature_2"].values)


def test_B12_e_hat_initial_is_field_aligned(tmp_path):
    dt = beam_me_up(**synthetic(tmp_path))
    e_hat_initial = dt["solver"].ds["e_hat_initial"]
    assert e_hat_initial.dims == ("col_field",)
    assert list(e_hat_initial["col_field"].values) == ["u1", "u2", "b"]


@pytest.mark.parametrize(
    "name",
    [
        "epsilon_bb",
        "epsilon_11",
        "epsilon_12",
        "H_Cardano_1",
        "H_eigval1",
        "e_eigvec1",
        "q_X_output",
        "K_Y_output",
    ],
)
def test_duplicated_outputs_removed(tmp_path, name):
    dt = beam_me_up(**synthetic(tmp_path))
    assert all(name not in node.ds for node in dt.subtree)


# ---------------------------------------------------------------- boundary conditions


@pytest.mark.parametrize("Psi_BC_flag", ["continuous", None])
def test_B2_density_at_entry_refused(tmp_path, Psi_BC_flag):
    """The density is not zero where the plasma starts (psi = 1), so keeping
    the vacuum K would put the ray off the dispersion relation"""
    kwargs = synthetic(
        tmp_path,
        Psi_BC_flag=Psi_BC_flag,
        density_fit_method=QuadraticFit(1.05, 4.0),
        poloidal_flux_enter=1.0,
    )
    # The density falls to zero further out, with a kink, so both remedies apply
    with pytest.raises(
        ValueError,
        match="'discontinuous' with `poloidal_flux_enter` = 1.04.*"
        "'continuous' with `poloidal_flux_enter` = 1.05",
    ):
        beam_me_up(**kwargs)


@pytest.mark.parametrize(("ratio", "refused"), [(3e-5, True), (3e-6, False)])
def test_B2_tolerance(tmp_path, ratio, refused):
    """The density left at the entry point, relative to its maximum, just above
    and just below the tolerance (1e-5). The profile is cut at the entry point,
    so the check must look just inside it, not only at the entry point found by
    root finding, which can land on either side of the jump"""
    kwargs = synthetic(
        tmp_path,
        Psi_BC_flag="continuous",
        density_fit_method=PolynomialFit(1.0, -4.0 * (1 - ratio), 0.0, 4.0),
    )
    if refused:
        # The profile jumps at the entry point: only "discontinuous" helps
        with pytest.raises(
            ValueError, match="'discontinuous' with `poloidal_flux_enter` = 0.99"
        ) as error:
            beam_me_up(**kwargs)
        assert "'continuous' with" not in str(error.value)
    else:
        dt = beam_me_up(**kwargs)
        assert np.max(np.abs(dt["beam"].ds["H_Booker"].values)) < 1e-4


def test_B2_discontinuous_accepts_density_at_entry(tmp_path):
    dt = beam_me_up(
        **synthetic(
            tmp_path,
            Psi_BC_flag="discontinuous",
            density_fit_method=QuadraticFit(1.05, 4.0),
            poloidal_flux_enter=1.0,
        )
    )
    # The usual drift at the default tolerances, not the 5e-2 the continuous
    # condition gave here
    assert np.max(np.abs(dt["beam"].ds["H_Booker"].values)) < 1e-4


# ---------------------------------------------------------------- inputs


def test_B9_the_profiles_zero_point_is_used_and_recorded(tmp_path):
    """The profile's own zero point is what the run uses, and the output
    records it, with the entry point where the density starts"""
    dt = beam_me_up(**synthetic(tmp_path, density_fit_method=QuadraticFit(1.05, 4.0)))
    assert float(dt["inputs"].ds["poloidal_flux_zero_density"]) == 1.05
    assert float(dt["inputs"].ds["poloidal_flux_enter"]) == pytest.approx(
        1.05, abs=1e-9
    )


@pytest.mark.parametrize(
    "name", ["poloidal_flux_zero_density", "poloidal_flux_zero_temperature"]
)
def test_zero_points_deprecated(tmp_path, name):
    """Still accepted, with their old meaning, but each warns"""
    with pytest.warns(FutureWarning, match=f"`{name}` is deprecated"):
        beam_me_up(**synthetic(tmp_path, **{name: 1.0}))


def test_no_deprecation_warning_without_the_zero_points(tmp_path):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        beam_me_up(**synthetic(tmp_path))


def test_named_fit_keeps_its_parameters(tmp_path):
    """A named fit's parameters mean what they did: the zero point is not
    among them"""
    with pytest.warns(FutureWarning):
        dt = beam_me_up(
            **synthetic(
                tmp_path,
                density_fit_method="quadratic",
                density_fit_parameters=[4.0],
                poloidal_flux_zero_density=1.05,
            )
        )
    assert float(dt["inputs"].ds["poloidal_flux_zero_density"]) == 1.05
    assert float(dt["inputs"].ds["poloidal_flux_enter"]) == pytest.approx(
        1.05, abs=1e-9
    )


def test_B10_preset_frequency_given_wins():
    assert get_parameters_for_Scotty("DBS_synthetic", 40.0)["launch_freq_GHz"] == 40.0
    assert get_parameters_for_Scotty("DBS_synthetic")["launch_freq_GHz"] == 55.0


def test_B10_preset_without_frequency_raises():
    with pytest.raises(ValueError, match="needs `launch_freq_GHz`"):
        get_parameters_for_Scotty("DBS_NSTX_MAST")


@pytest.mark.parametrize("name", ["launch_beam_width", "launch_beam_curvature"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_launch_beam_not_finite_raises(tmp_path, name, value):
    with pytest.raises(ValueError, match="must be finite|must be positive"):
        beam_me_up(**synthetic(tmp_path, **{name: value}))


@pytest.mark.parametrize("name", ["further_analysis_flag", "detailed_analysis_flag"])
def test_B22_flags_removed(name):
    assert name not in inspect.signature(beam_me_up).parameters
    assert name not in from_v3_kwargs({name: True})


def test_figure_flag_kept():
    """`figure_flag` stays (supervisor, 14 Nov), for figures `beam_me_up` may
    make later; v3 scripts keep it"""
    assert inspect.signature(beam_me_up).parameters["figure_flag"].default is False
    assert from_v3_kwargs({"figure_flag": True})["figure_flag"] is True


# ---------------------------------------------------------------- v3 scripts


def test_converted_v3_script_enters_where_the_density_is_zero(tmp_path):
    """v3's defaults: density zero at 1.01. The plasma now starts where the
    density does, there, with
    an analytic vacuum launch and the continuous condition, and the ray stays
    on the dispersion relation"""
    v3_script = synthetic(tmp_path)
    for name in (
        "launch_flag",
        "Psi_BC_flag",
        "geometry",
        "len_tau",
    ):
        del v3_script[name]
    v3_script.update(density_fit_method="quadratic", density_fit_parameters=[4.0])
    kwargs = from_v3_kwargs(v3_script)
    assert kwargs["launch_flag"] == "vacuum_analytic"
    assert kwargs["Psi_BC_flag"] == "continuous"
    assert "poloidal_flux_enter" not in kwargs
    assert kwargs["poloidal_flux_zero_density"] == 1.01
    with pytest.warns(FutureWarning):
        dt = beam_me_up(**kwargs)
    assert float(dt["inputs"].ds["poloidal_flux_enter"]) == pytest.approx(
        1.01, abs=1e-9
    )
    # Entering at 1.0 instead, as before, gave |H| = 1e-2 all along the ray. The
    # bound allows for v3's loose tolerances (rtol 1e-3), which from_v3_kwargs sets
    assert np.max(np.abs(dt["beam"].ds["H_Booker"].values)) < 1e-4


# ---------------------------------------------------------------- B7: launch angles


@pytest.mark.parametrize("zeta", [0.3, -1.2, 2.5])
def test_B7_launch_direction_turns_with_the_antenna(zeta):
    angles = (6.0, -5.0)
    K_hat_0 = launch_direction_cartesian(2.5, 0.0, *angles)
    K_hat = launch_direction_cartesian(2.5 * np.cos(zeta), 2.5 * np.sin(zeta), *angles)
    rotation = np.array(
        [
            [np.cos(zeta), -np.sin(zeta), 0],
            [np.sin(zeta), np.cos(zeta), 0],
            [0, 0, 1],
        ]
    )
    assert_allclose(K_hat, rotation @ K_hat_0, atol=1e-15)
    # ray_line moves along the same direction
    q = np.array([2.5 * np.cos(zeta), 2.5 * np.sin(zeta), 0.1])
    assert_allclose(ray_line(*q, 0.7, *angles), q + 0.7 * K_hat, atol=1e-15)


@pytest.mark.parametrize("Psi_BC_flag", ["continuous", "discontinuous"])
def test_B7_antenna_at_zeta_gives_the_rotated_run(tmp_path, Psi_BC_flag):
    """In an axisymmetric field, the same angles from an antenna at zeta_0
    give the same beam, turned by zeta_0: the cylindrical components are
    unchanged, and q_zeta is shifted by zeta_0. The runs differ only by
    integration error, which falls with rtol (Kai, 2026-11-14: q differs by
    7e-6 m at rtol 1e-4, 7e-9 m at 1e-7 and 2e-11 m at 1e-9), so they are
    run at rtol 1e-7 and compared at about ten times that difference"""
    zeta_0 = 0.3
    runs = []
    for zeta in (0.0, zeta_0):
        kwargs = synthetic(
            tmp_path / f"zeta{zeta}",
            Psi_BC_flag=Psi_BC_flag,
            toroidal_launch_angle_Torbeam=4.0,
            rtol=1e-7,
            atol=1e-10,
        )
        kwargs["launch_position"] = np.array([2.587, zeta, -0.0157])
        runs.append(beam_me_up(**kwargs)["solver"].ds)
    ref, turned = runs
    for name in ("q_R_output", "q_Z_output"):
        assert_allclose(turned[name], ref[name], rtol=0, atol=1e-7)
    assert_allclose(
        turned["q_zeta_output"], ref["q_zeta_output"] + zeta_0, rtol=0, atol=1e-7
    )
    for name in ("K_R_output", "K_zeta_output", "K_Z_output"):
        assert_allclose(turned[name], ref[name], rtol=0, atol=1e-3)
    Psi = ref["Psi_3D_output_labframe_cylindrical"].values
    assert_allclose(
        turned["Psi_3D_output_labframe_cylindrical"].values,
        Psi,
        rtol=0,
        atol=1e-4 * np.abs(Psi).max(),
    )


# ------------------------------------------- where the density starts (Eduard)

PSI_DATA = np.linspace(0.0, 1.0, 101) ** 2  # the docstring's ne.dat: rho 0 to 1


def kinked_file_profile(zero_point=1.0):
    """n_e = 4(1 - psi), as in the `from_dat_file` docstring: zero at the end
    of its data, with a kink there"""
    return SmoothingSplineFit(zero_point, PSI_DATA, 4.0 * (1 - PSI_DATA))


def flat_file_profile(zero_point=1.0):
    """n_e = 4(1 - psi)^2: zero and flat at the end of its data"""
    return SmoothingSplineFit(zero_point, PSI_DATA, 4.0 * (1 - PSI_DATA) ** 2)


def test_density_start_inside_a_zero_band():
    """Between the end of the data (1.0) and the zero point (1.01), the spline
    holds its last value, 0"""
    from scotty.boundary_conditions import find_density_start

    profile = kinked_file_profile(1.01)
    assert profile(1.005) == 0.0
    assert find_density_start(profile, 1.01, 1e-10 * 4.0) == pytest.approx(
        1.0, abs=1e-6
    )


@pytest.mark.parametrize("Psi_BC_flag", ["continuous", "discontinuous", None])
def test_kink_inside_the_entry_point_refused(tmp_path, Psi_BC_flag):
    """Eduard's case 1: the density is zero from the end of its data (1.0) to
    the entry point (1.01), so the beam crosses the kink at 1.0 with no
    boundary condition. Every condition refuses it, and gives both remedies:
    'discontinuous' slightly inside, or 'continuous' at the kink"""
    kwargs = synthetic(
        tmp_path,
        Psi_BC_flag=Psi_BC_flag,
        density_fit_method=kinked_file_profile(1.01),
        poloidal_flux_enter=1.01,
    )
    with pytest.raises(
        ValueError,
        match="starts with a kink at 1, .*'discontinuous' with `poloidal_flux_enter` "
        "= 0.99.*'continuous' with `poloidal_flux_enter` = 1 ",
    ):
        beam_me_up(**kwargs)


def test_zero_inside_the_plasma_refused(tmp_path):
    """A density that is zero in a band inside where it starts, as a spline
    through zero-padded data can give: the beam would cross it"""

    def density(psi):
        psi = np.asarray(psi, dtype=float)
        return 4.0 * (1 - psi) * (np.abs(psi - 0.5) > 0.05)

    kwargs = synthetic(
        tmp_path,
        Psi_BC_flag="continuous",
        density_fit_method=density,
        poloidal_flux_enter=1.0,
    )
    with pytest.raises(
        ValueError, match="inside the plasma, between poloidal flux 0.45"
    ):
        beam_me_up(**kwargs)


def test_discontinuous_slightly_inside_the_kink_agrees(tmp_path):
    """The supervisor's simple recipe: 'discontinuous' 0.01 inside the kink, at
    the default tolerance, against 'continuous' at the kink, converged. The
    layer left outside (n_e up to 1% of the axis value) changes the width at
    the cut-off by less than 1e-3 (Kai, 2026-11-15: 0.01% for entries from
    0.999 to 0.9)"""
    widths = []
    for flag, enter, rtol in (
        ("discontinuous", 0.99, 1e-4),
        ("continuous", 1.0, 1e-7),
    ):
        dt = beam_me_up(
            **synthetic(
                tmp_path / flag,
                Psi_BC_flag=flag,
                density_fit_method=kinked_file_profile(1.01),
                poloidal_flux_enter=enter,
                rtol=rtol,
                atol=rtol * 1e-3,
            )
        )
        cutoff = int(np.argmin(dt["solver"].ds["K_output_magnitude"].values))
        widths.append(float(dt["beam"].ds["width_1"].values[cutoff]))
    assert widths[0] == pytest.approx(widths[1], rel=1e-3)


def test_flat_start_inside_the_entry_point_accepted(tmp_path):
    """A zero band is fine where the density starts flat: the solver crosses
    nothing it can't integrate"""
    dt = beam_me_up(
        **synthetic(
            tmp_path,
            density_fit_method=flat_file_profile(1.01),
            Psi_BC_flag=None,
            poloidal_flux_enter=1.01,
        )
    )
    assert float(dt["inputs"].ds["poloidal_flux_enter"]) == 1.01


def test_None_with_a_kink_at_the_entry_point_refused(tmp_path):
    """Eduard's case 2: entering at the kink (1.0) with no boundary condition.
    Before, only the zero point (1.01) was checked, where the profile is flat"""
    kwargs = synthetic(
        tmp_path,
        Psi_BC_flag=None,
        density_fit_method=kinked_file_profile(1.01),
        poloidal_flux_enter=1.0,
    )
    with pytest.raises(ValueError, match="it is not flat"):
        beam_me_up(**kwargs)


def test_default_entry_with_a_file_is_converged(tmp_path):
    """With the entry point left unset, the plasma starts where the density
    does (the end of the file's data, inside the zero point 1.01), with the
    boundary condition there: the width at the cut-off
    at v3's loose tolerance agrees with a tight run (Eduard's case 1 gave 0.107
    m against 0.0498 m when entering at 1.01)"""
    widths = []
    for rtol in (1e-3, 1e-7):
        dt = beam_me_up(
            **synthetic(
                tmp_path / f"rtol{rtol}",
                density_fit_method=kinked_file_profile(1.01),
                rtol=rtol,
                atol=rtol * 1e-3,
            )
        )
        beam = dt["beam"].ds
        cutoff = int(np.argmin(dt["solver"].ds["K_output_magnitude"].values))
        widths.append(float(beam["width_1"].values[cutoff]))
        assert float(dt["inputs"].ds["poloidal_flux_enter"]) == pytest.approx(
            1.0, abs=1e-6
        )
    assert widths[0] == pytest.approx(widths[1], rel=1e-2)


def width_at_cutoff(dt):
    cutoff = int(np.argmin(dt["solver"].ds["K_output_magnitude"].values))
    return float(dt["beam"].ds["width_1"].values[cutoff])


def plateau_file_profile():
    """Data that end at a finite density (0.3) inside the zero point (1.01):
    the spline holds 0.3 beyond them, with a kink at the end of the data"""
    return SmoothingSplineFit(1.01, PSI_DATA, 4.0 * (1 - PSI_DATA) + 0.3)


@pytest.fixture(scope="module")
def plateau_converged(tmp_path_factory):
    """Entering at the end of the data, at a tight tolerance (the referee,
    2026-11-15: 0.04542 m; entering at 1.01 gave 0.1127 m at rtol 1e-4)"""
    return width_at_cutoff(
        beam_me_up(
            **synthetic(
                tmp_path_factory.mktemp("plateau"),
                Psi_BC_flag="discontinuous",
                density_fit_method=plateau_file_profile(),
                poloidal_flux_enter=1.0,
                rtol=1e-7,
                atol=1e-10,
            )
        )
    )


def test_default_entry_stops_at_the_end_of_the_data(tmp_path, plateau_converged):
    dt = beam_me_up(
        **synthetic(
            tmp_path,
            Psi_BC_flag="discontinuous",
            density_fit_method=plateau_file_profile(),
        )
    )
    assert float(dt["inputs"].ds["poloidal_flux_enter"]) == 1.0
    assert width_at_cutoff(dt) == pytest.approx(plateau_converged, rel=1e-3)


def test_entry_beyond_the_end_of_the_data_refused(tmp_path):
    kwargs = synthetic(
        tmp_path,
        Psi_BC_flag="discontinuous",
        density_fit_method=plateau_file_profile(),
        poloidal_flux_enter=1.01,
    )
    with pytest.raises(
        ValueError,
        match="data end at poloidal flux 1, .*kink at 1 .*'discontinuous' with "
        "`poloidal_flux_enter` = 0.99",
    ):
        beam_me_up(**kwargs)


def test_continuous_at_a_plateau_advises_the_end_of_the_data(tmp_path):
    """Refused (the density is finite at the entry point), and the advice
    must not send the beam out past the end of the data, to 1.01"""
    kwargs = synthetic(
        tmp_path,
        Psi_BC_flag="continuous",
        density_fit_method=plateau_file_profile(),
    )
    with pytest.raises(ValueError, match="'discontinuous'") as error:
        beam_me_up(**kwargs)
    assert "1.01" not in str(error.value).split("keeps the vacuum K")[1]


def test_flat_data_end_inside_the_entry_accepted(tmp_path):
    """Data that end flat (here at a finite density) carry straight on into
    the held value: no kink, nothing to refuse"""
    psi = np.linspace(0.0, 1.0, 101) ** 2
    profile = SmoothingSplineFit(1.01, psi, 0.3 + 4.0 * (1 - psi) ** 2)
    beam_me_up(
        **synthetic(
            tmp_path,
            Psi_BC_flag="discontinuous",
            density_fit_method=profile,
            poloidal_flux_enter=1.01,
        )
    )


def test_default_entry_moves_inside_a_kink_not_at_1(tmp_path):
    """The data end, with a kink at n_e = 0, at 0.95, inside the zero point
    1.0. The old default entry, 1.0, crossed the kink; the new one starts at
    it, and agrees with a tight run at v3's loose tolerance"""
    psi = np.linspace(0.0, 0.95, 101)
    profile = SmoothingSplineFit(1.0, psi, 4.0 * (0.95 - psi) / 0.95)
    widths = []
    for rtol in (1e-3, 1e-7):
        dt = beam_me_up(
            **synthetic(
                tmp_path / f"rtol{rtol}",
                density_fit_method=profile,
                rtol=rtol,
                atol=rtol * 1e-3,
            )
        )
        assert float(dt["inputs"].ds["poloidal_flux_enter"]) == pytest.approx(
            0.95, abs=1e-6
        )
        widths.append(width_at_cutoff(dt))
    assert widths[0] == pytest.approx(widths[1], rel=1e-2)


@pytest.mark.parametrize("enter", [None, 0.99])
def test_jump_with_discontinuous(tmp_path, enter):
    """A density that jumps at its zero point, entering at the jump (left
    unset) or slightly inside it, both stay on the dispersion surface"""
    dt = beam_me_up(
        **synthetic(
            tmp_path,
            Psi_BC_flag="discontinuous",
            # n_e = 4 - 3.6 psi^2: 0.4 at its zero point, 1.0
            density_fit_method=PolynomialFit(1.0, -3.6, 0.0, 4.0),
            **({} if enter is None else {"poloidal_flux_enter": enter}),
        )
    )
    assert np.max(np.abs(dt["beam"].ds["H_Booker"].values)) < 1e-4


def test_temperature_zero_inside_the_default_entry_refused(tmp_path):
    """The temperature checks also run once the entry point is set from the
    density (the referee, 2026-11-15)"""
    kwargs = synthetic(
        tmp_path,
        density_fit_method=QuadraticFit(1.05, 4.0),
        relativistic_flag=True,
        temperature_fit_method=QuadraticFit(1.0, 1.0),
    )
    with pytest.raises(ValueError, match="temperature would be zero inside"):
        beam_me_up(**kwargs)


def test_unused_temperature_not_checked(tmp_path):
    """Without relativity the temperature profile isn't used, so its zero
    point doesn't matter (the referee, 2026-11-15)"""
    beam_me_up(
        **synthetic(
            tmp_path,
            density_fit_method=QuadraticFit(1.05, 4.0),
            relativistic_flag=False,
            temperature_fit_method=QuadraticFit(1.0, 1.0),
        )
    )


# ---------------------------------------------------------------- presets


def test_presets_take_one_data_folder(tmp_path):
    import inspect

    parameters = inspect.signature(get_parameters_for_Scotty).parameters
    assert "user" not in parameters
    kwargs = get_parameters_for_Scotty(
        "DBS_UCLA_MAST-U", 55.0, find_B_method="torbeam", data_path=tmp_path
    )
    assert kwargs["ne_data_path"] == tmp_path
    assert kwargs["magnetic_data_path"] == tmp_path


def test_v3_plain_function_keeps_v3s_entry():
    """A plain function has no zero point, so Scotty can't tell where its
    density starts: the converted script enters at v3's 1.0"""
    kw = from_v3_kwargs({"density_fit_method": lambda psi: 4.0 * (1 - psi)})
    assert kw["poloidal_flux_enter"] == 1.0


def test_v3_plasma_launch_keeps_v3s_entry():
    """There the entry point is only where the ray stops"""
    kw = from_v3_kwargs({"vacuumLaunch_flag": False})
    assert kw["launch_flag"] == "plasma"
    assert kw["poloidal_flux_enter"] == 1.0


def test_v3_given_entry_kept():
    kw = from_v3_kwargs({"poloidal_flux_enter": 0.98})
    assert kw["poloidal_flux_enter"] == 0.98


def test_synthetic_preset_leaves_the_zero_point_to_its_profile():
    kwargs = get_parameters_for_Scotty("DBS_synthetic")
    for name in (
        "poloidal_flux_zero_density",
        "poloidal_flux_zero_temperature",
        "poloidal_flux_enter",
    ):
        assert name not in kwargs
