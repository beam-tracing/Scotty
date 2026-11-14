"""Tests of launching from inside the plasma (``launch_flag="plasma"``) and of
launching from vacuum without boundary conditions (``Psi_BC_flag=None``).

The main check is a round trip: a plasma launch from a point on a vacuum
launch's beam, with the beam's K and Psi there, gives the same beam. This is
done from the plasma edge, where the vacuum launch's solvers start, and from
a point inside the plasma, where K tells the O- and X-modes apart (at the
edge the density is zero, so both dispersion relations hold).

Both runs use a tight solver tolerance, since the starting values are only
equal to rounding and the adaptive steps otherwise amplify that to the
solver's tolerance. From inside, the beams are compared at matched tau; their
last points differ slightly because each run stops at its own ray solver's
exit tau.

Only cylindrical geometry is tested end to end: there is no Cartesian field
source to run ``beam_me_up`` on.
"""

from scotty.beam_me_up import beam_me_up
from scotty.boundary_conditions import check_density_smooth_at_edge
from scotty.compat import from_v3_kwargs
from scotty.init_bruv import get_parameters_for_Scotty
from scotty.profile_fit import PolynomialFit, QuadraticFit

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.interpolate import CubicSpline

# Tight solver tolerances for comparing two runs: the round trip then agrees
# to ~3e-7 in ~3 s per run (tighter tolerances are much slower)
TIGHT = {"rtol": 1e-6, "atol": 1e-9}

BEAM = [
    "q_output_cylindrical",
    "K_output_cylindrical",
    "Psi_3D_output_labframe_cylindrical",
]


def synthetic(tmp_path, **changes):
    kwargs = get_parameters_for_Scotty("DBS_synthetic")
    kwargs.update(len_tau=10, output_path=tmp_path)
    kwargs.update(changes)
    return kwargs


def plasma_launch_from(vacuum_run, kwargs, index=None, **changes):
    """Plasma-launch arguments that start where `vacuum_run`'s solvers do
    (`index` None), or at its `index`-th output point"""
    so = vacuum_run["solver"]
    if index is None:
        q = so["q_initial_cylindrical"].values
        K = so["K_initial_cartesian"].values
        Psi = so["Psi_3D_initial_labframe_cartesian"].values
    else:
        q = so["q_output_cylindrical"].values[index]
        K = so["K_output_cartesian"].values[index]
        Psi = so["Psi_3D_output_labframe_cartesian"].values[index]
    plasma = dict(kwargs)
    plasma.update(
        launch_flag="plasma",
        launch_position=q,
        plasmaLaunch_K_cartesian=K,
        plasmaLaunch_Psi_3D_lab_cartesian=Psi,
    )
    plasma.update(changes)
    return plasma


##################################################
# Plasma launch from the plasma edge
##################################################


@pytest.fixture(
    scope="module",
    params=[(0.0, 1), (10.0, 1), (10.0, -1)],
    ids=["mode+1-tor0", "mode+1-tor10", "mode-1-tor10"],
)
def round_trip(request, tmp_path_factory):
    toroidal_angle, mode_flag = request.param
    kwargs = synthetic(
        tmp_path_factory.mktemp("plasma_launch"),
        toroidal_launch_angle_Torbeam=toroidal_angle,
        mode_flag=mode_flag,
        **TIGHT,
    )
    vacuum = beam_me_up(**kwargs)
    plasma = beam_me_up(**plasma_launch_from(vacuum, kwargs))
    return vacuum, plasma


@pytest.mark.parametrize("name", BEAM)
def test_plasma_launch_reproduces_the_vacuum_beam(round_trip, name):
    vacuum, plasma = round_trip
    a = vacuum["solver"][name].values
    b = plasma["solver"][name].values

    assert_allclose(b, a, rtol=1e-5, atol=1e-5 * np.abs(a).max())


def test_plasma_launch_keeps_the_mode_and_tau(round_trip):
    """At the plasma edge the density is zero, so both runs save the mode's
    zero-density polarisation as e_hat_initial"""
    vacuum, plasma = round_trip
    so_v, so_p = vacuum["solver"], plasma["solver"]

    assert int(so_p["mode_flag_initial"]) == int(so_v["mode_flag_initial"])
    assert_allclose(so_p["tau"].values, so_v["tau"].values, rtol=1e-6)
    assert_allclose(
        np.abs(so_p["e_hat_initial"].values),
        np.abs(so_v["e_hat_initial"].values),
        atol=1e-6,
    )


def test_plasma_launch_starts_at_the_launch_point(round_trip):
    """Launch, entry and initial values are the same, and the distance from
    launch to entry is zero"""
    _, plasma = round_trip
    so = plasma["solver"]

    assert float(so["distance_from_launch_to_entry"]) == 0.0
    assert_allclose(so["q_initial_cylindrical"], so["q_launch_cylindrical"])
    assert_allclose(so["K_initial_cylindrical"], so["K_launch_cylindrical"])
    assert_allclose(
        so["Psi_3D_initial_labframe_cylindrical"],
        so["Psi_3D_launch_labframe_cylindrical"],
    )


def test_plasma_launch_inputs_are_saved(round_trip):
    vacuum, plasma = round_trip
    inputs, so_v = plasma["inputs"], vacuum["solver"]

    assert str(inputs["launch_flag"].values) == "plasma"
    assert_allclose(inputs["plasmaLaunch_K_cartesian"], so_v["K_initial_cartesian"])
    assert_allclose(
        inputs["plasmaLaunch_Psi_3D_lab_cartesian"],
        so_v["Psi_3D_initial_labframe_cartesian"],
    )


def test_plasma_launch_has_no_beam_waist(round_trip):
    """filterfun_b is normalised by the launched beam's waist, which a plasma
    launch doesn't have"""
    _, plasma = round_trip
    dbs = plasma["diagnostics/dbs"]

    assert np.isnan(float(plasma["beam"]["beam_waist"]))
    assert np.isnan(dbs["filterfun_b"].values).all()


@pytest.mark.parametrize(("named", "mode_flag"), [("O", 1), ("X", -1)])
def test_plasma_launch_at_zero_density_with_named_mode(tmp_path, named, mode_flag):
    """At the edge, K satisfies both modes' dispersion relations, but their
    zero-density polarisations still tell them apart. For DBS_synthetic the
    O-mode is mode_flag = 1 (test_vacuum_mode_flag_polarisation)"""
    kwargs = synthetic(tmp_path)
    vacuum = beam_me_up(**kwargs)
    plasma = beam_me_up(**plasma_launch_from(vacuum, kwargs, mode_flag=named))

    assert int(plasma["solver"]["mode_flag_initial"]) == mode_flag


##################################################
# Plasma launch from inside the plasma
##################################################

# Output point of the vacuum run to launch from: about 20% of the way along,
# where the density is about half its peak
LEN_TAU_INSIDE = 400
INSIDE = LEN_TAU_INSIDE // 5


@pytest.fixture(scope="module", params=[(1, "O"), (-1, "X")], ids=["O", "X"])
def inside(request, tmp_path_factory):
    """A vacuum launch with numeric `mode_flag`, densely sampled, and its
    plasma launch from inside the plasma. For DBS_synthetic, mode_flag = 1
    is the O-mode (test_vacuum_mode_flag_polarisation checks this)"""
    mode_flag, mode_name = request.param
    kwargs = synthetic(
        tmp_path_factory.mktemp("inside"),
        mode_flag=mode_flag,
        toroidal_launch_angle_Torbeam=10.0,
        len_tau=LEN_TAU_INSIDE,
        **TIGHT,
    )
    vacuum = beam_me_up(**kwargs)
    plasma = beam_me_up(
        **plasma_launch_from(vacuum, kwargs, index=INSIDE, mode_flag=mode_flag)
    )
    return mode_name, mode_flag, kwargs, vacuum, plasma


def test_vacuum_mode_flag_polarisation(inside):
    """Inside the plasma the O-mode's polarisation is along b (|e_b| near 1)"""
    mode_name, _, _, vacuum, _ = inside
    e_b = np.abs(vacuum["beam"]["e_hat"].values[INSIDE, -1])

    assert (e_b > 0.75) == (mode_name == "O")


@pytest.mark.parametrize(
    ("name", "tolerance"),
    [
        (BEAM[0], 1e-5),
        (BEAM[1], 1e-5),
        # Psi changes quickly near the cut-off, so the spline through the
        # vacuum run's samples is only good to ~3e-4 there at this len_tau
        (BEAM[2], 1e-3),
    ],
)
def test_plasma_launch_from_inside_follows_the_vacuum_beam(inside, name, tolerance):
    _, _, _, vacuum, plasma = inside
    so_v, so_p = vacuum["solver"], plasma["solver"]
    tau_v = so_v["tau"].values
    tau_p = so_p["tau"].values + tau_v[INSIDE]
    overlap = tau_p <= tau_v[-1]

    reference = CubicSpline(tau_v, so_v[name].values, axis=0)(tau_p[overlap])
    values = so_p[name].values[overlap]
    assert_allclose(
        values,
        reference,
        rtol=tolerance,
        atol=tolerance * np.abs(so_v[name].values).max(),
    )


def test_plasma_launch_from_inside_finds_the_named_mode(inside):
    """From inside, "O" or "X" is found from K and its polarisation, and
    gives the vacuum run's mode"""
    mode_name, mode_flag, kwargs, vacuum, _ = inside
    plasma = beam_me_up(
        **plasma_launch_from(vacuum, kwargs, index=INSIDE, mode_flag=mode_name)
    )

    assert int(plasma["solver"]["mode_flag_initial"]) == mode_flag


def test_plasma_launch_from_inside_rejects_the_other_named_mode(inside):
    mode_name, _, kwargs, vacuum, _ = inside
    other = "X" if mode_name == "O" else "O"

    with pytest.raises(ValueError, match=f"K is the {mode_name}-mode"):
        beam_me_up(**plasma_launch_from(vacuum, kwargs, index=INSIDE, mode_flag=other))


def test_plasma_launch_from_inside_rejects_the_other_mode_flag(inside):
    """K of one mode with the other mode's `mode_flag` is caught, not traced"""
    _, mode_flag, kwargs, vacuum, _ = inside

    with pytest.raises(ValueError, match="satisfies the dispersion relation of"):
        beam_me_up(
            **plasma_launch_from(vacuum, kwargs, index=INSIDE, mode_flag=-mode_flag)
        )


##################################################
# Plasma launch: other options and errors
##################################################


def test_plasma_launch_ray_tracing_only(tmp_path):
    """With ray_tracing_flag, the ray starts from K alone: the Psi given (which
    the input checks still require) is not used"""
    kwargs = synthetic(tmp_path, ray_tracing_flag=True, **TIGHT)
    vacuum = beam_me_up(**kwargs)
    so = vacuum["solver"]
    plasma = beam_me_up(
        **dict(
            kwargs,
            launch_flag="plasma",
            launch_position=so["q_initial_cylindrical"].values,
            plasmaLaunch_K_cartesian=so["K_initial_cartesian"].values,
            plasmaLaunch_Psi_3D_lab_cartesian=np.eye(3),
        )
    )

    assert_allclose(
        plasma["solver"]["q_output_cylindrical"],
        so["q_output_cylindrical"],
        rtol=1e-6,
    )


def test_plasma_launch_through_compat(tmp_path):
    """A v3 plasma launch (cylindrical plasmaLaunch_K) runs through
    from_v3_kwargs and starts with that K"""
    kwargs = synthetic(tmp_path)
    vacuum = beam_me_up(**kwargs)
    so = vacuum["solver"]

    v3 = dict(kwargs)
    for name in ("geometry", "launch_flag", "Psi_BC_flag"):
        v3.pop(name)
    v3.update(
        vacuumLaunch_flag=False,
        Psi_BC_flag="continuous",
        launch_position=so["q_initial_cylindrical"].values,
        plasmaLaunch_K=so["K_initial_cylindrical"].values,
        plasmaLaunch_Psi_3D_lab_Cartesian=so[
            "Psi_3D_initial_labframe_cartesian"
        ].values,
    )
    plasma = beam_me_up(**from_v3_kwargs(v3))

    assert str(plasma["inputs"]["launch_flag"].values) == "plasma"
    assert_allclose(
        plasma["solver"]["K_initial_cylindrical"],
        so["K_initial_cylindrical"],
        rtol=1e-12,
    )


def test_plasma_launch_outside_the_plasma(tmp_path):
    kwargs = synthetic(tmp_path)
    vacuum = beam_me_up(**kwargs)
    # The antenna position is in vacuum
    plasma_kwargs = plasma_launch_from(
        vacuum, kwargs, launch_position=kwargs["launch_position"]
    )

    with pytest.raises(ValueError, match="outside the plasma"):
        beam_me_up(**plasma_kwargs)


def test_plasma_launch_K_off_the_dispersion_relation(tmp_path):
    kwargs = synthetic(tmp_path)
    vacuum = beam_me_up(**kwargs)
    K = vacuum["solver"]["K_initial_cartesian"].values
    plasma_kwargs = plasma_launch_from(vacuum, kwargs, plasmaLaunch_K_cartesian=1.1 * K)

    with pytest.raises(ValueError, match="does not satisfy the dispersion relation"):
        beam_me_up(**plasma_kwargs)


@pytest.mark.parametrize(
    "missing", ["plasmaLaunch_K_cartesian", "plasmaLaunch_Psi_3D_lab_cartesian"]
)
def test_plasma_launch_needs_K_and_Psi(tmp_path, missing):
    kwargs = synthetic(tmp_path)
    vacuum = beam_me_up(**kwargs)
    plasma_kwargs = plasma_launch_from(vacuum, kwargs)
    plasma_kwargs.pop(missing)

    with pytest.raises(ValueError, match="plasmaLaunch"):
        beam_me_up(**plasma_kwargs)


##################################################
# Psi_BC_flag = None
##################################################

# n_e = 4 (1 - psi^2)^2: zero and flat at the edge psi = 1. (The simpler
# 4 (1 - psi)^2 stays so low near the edge that analysis can't separate the
# O- and X-mode Cardano roots, with or without boundary conditions.)
FLAT_AT_EDGE = PolynomialFit(1.0, 4.0, 0.0, -8.0, 0.0, 4.0)


@pytest.fixture(scope="module")
def no_boundary_condition(tmp_path_factory):
    kwargs = synthetic(
        tmp_path_factory.mktemp("no_bc"), density_fit_method=FLAT_AT_EDGE, **TIGHT
    )
    without = beam_me_up(**dict(kwargs, Psi_BC_flag=None))
    continuous = beam_me_up(**dict(kwargs, Psi_BC_flag="continuous"))
    return without, continuous


def test_no_boundary_condition_carries_K_and_Psi_over(no_boundary_condition):
    """K and Psi where the solvers start are the vacuum ones at the entry"""
    without, _ = no_boundary_condition
    so = without["solver"]

    assert str(without["inputs"]["Psi_BC_flag"].values) == "None"
    assert_allclose(so["K_initial_cartesian"], so["K_launch_cartesian"], rtol=1e-12)
    assert_allclose(
        so["Psi_3D_initial_labframe_cylindrical"],
        so["Psi_3D_entry_labframe_cylindrical"],
    )


@pytest.mark.parametrize("name", BEAM)
def test_no_boundary_condition_matches_continuous_when_flat(
    no_boundary_condition, name
):
    """With density zero and flat at the edge, the continuous boundary
    condition has nothing to correct, so both give the same beam"""
    without, continuous = no_boundary_condition
    a = continuous["solver"][name].values
    b = without["solver"][name].values

    assert_allclose(b, a, rtol=1e-4, atol=1e-4 * np.abs(a).max())


@pytest.mark.parametrize(("named", "mode_flag"), [("O", 1), ("X", -1)])
def test_no_boundary_condition_with_named_mode(tmp_path, named, mode_flag):
    """The beam enters where the density is zero, where the modes are told
    apart by their zero-density polarisations"""
    kwargs = synthetic(
        tmp_path, density_fit_method=FLAT_AT_EDGE, Psi_BC_flag=None, mode_flag=named
    )
    dt = beam_me_up(**kwargs)

    assert int(dt["solver"]["mode_flag_initial"]) == mode_flag


def test_no_boundary_condition_rejects_a_kink(tmp_path):
    """DBS_synthetic's quadratic profile goes to zero at the edge with a
    finite gradient"""
    with pytest.raises(ValueError, match="not flat"):
        beam_me_up(**synthetic(tmp_path, Psi_BC_flag=None))


def test_no_boundary_condition_rejects_a_jump(tmp_path):
    """The profile is cut to zero at psi = 1, where it is still finite. The
    density at the entry point is then not zero either, which is refused first"""
    kwargs = synthetic(
        tmp_path,
        Psi_BC_flag=None,
        density_fit_method=PolynomialFit(1.0, -4.0 / 1.3, 0.0, 4.0),
        poloidal_flux_zero_density=1.0,
    )
    with pytest.raises(ValueError, match="density at the plasma entry point"):
        beam_me_up(**kwargs)


@pytest.mark.parametrize(
    ("profile", "problem"),
    [
        pytest.param(FLAT_AT_EDGE, None, id="flat"),
        pytest.param(PolynomialFit(1.0, -4.0, 4.0), "not flat", id="linear"),
        pytest.param(PolynomialFit(1.0, 4.0, -8.0, 4.1), "not continuous", id="offset"),
        pytest.param(PolynomialFit(1.0, 0.0), None, id="no-density"),
    ],
)
def test_check_density_smooth_at_edge(profile, problem):
    if problem is None:
        check_density_smooth_at_edge(profile, 1.0)
    else:
        with pytest.raises(ValueError, match=problem):
            check_density_smooth_at_edge(profile, 1.0)


def test_check_density_smooth_at_edge_tolerance():
    """A slope of 1e-4 of the maximum passes the default tolerance (1e-3)
    but not a tighter one"""
    nearly_flat = PolynomialFit(1.0, 4.0 + 4e-4, -8.0 - 4e-4, 4.0)
    check_density_smooth_at_edge(nearly_flat, 1.0)
    with pytest.raises(ValueError, match="not flat"):
        check_density_smooth_at_edge(nearly_flat, 1.0, rtol=1e-5)
