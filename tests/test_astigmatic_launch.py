"""Elliptical and astigmatic launch beams (`launch_beam_Psi_w`).

The reference is the vacuum propagation of a general Gaussian beam,
    Psi_w(s)^-1 = Psi_w(0)^-1 + s / K0 * 1,
in a fixed frame perpendicular to the (straight) ray. The launch beam frame is built here from its
definition (g_hat along K_launch; y_hat the upward unit vector perpendicular to g_hat in the vertical
plane through g_hat; x_hat = y_hat x g_hat), not from `scotty.launch.launch_beam_frame`, so that a
swapped or flipped axis fails these tests. The rotated astigmatic beam is what catches a flipped
x_hat: a beam whose axes are along x_hat and y_hat has Psi_xy = 0 and doesn't see its sign.

1. the Psi_w helper: principal widths and axes come back, and the circular limit holds;
2. the launch Psi_3D, projected on the frame, is the given Psi_w, with nothing along g;
3. `vacuum_analytic`: the entry Psi_3D is the analytic propagation over the launch-entry distance;
4. `vacuum_numerical`: the beam solver's own Psi along the vacuum part of the ray is the analytic one
   (an independent route: the ODE solver, and the cylindrical <-> Cartesian conversions of Psi);
5. through the plasma boundary, `vacuum_analytic` with the continuous BC (which changes Psi by a
   factor of ~3 at DBS_synthetic's density kink) agrees with `vacuum_numerical`, which integrates
   through the kink. Further in, the two routes differ by up to ~5e-4 relative, growing along the
   ray; master (ea3f00b) shows the same for a circular beam, so it is not from this change;
6. a circular beam given as Psi_w is bit-for-bit the run given as width and curvature;
7. bad inputs raise; the frame is right-handed and upward, also past the vertical.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from scotty.antenna.gaussian import (
    astigmatic_gaussian_Psi_w,
    circular_gaussian_Psi_w,
    principal_widths,
)
from scotty.beam_me_up import beam_me_up
from scotty.fun_general import freq_GHz_to_wavenumber
from scotty.init_bruv import get_parameters_for_Scotty

FREQ_GHZ = 55.0  # DBS_synthetic
K0 = freq_GHz_to_wavenumber(FREQ_GHZ)

BEAMS = {
    # widths along x and y, same curvature: elliptical, no astigmatism
    "elliptical": dict(widths=(0.03, 0.05), curvatures=(-0.25, -0.25)),
    # width axes at 30 deg, curvature axes at -50 deg, one converging and one diverging
    "rotated_astigmatic": dict(
        widths=(0.03, 0.05),
        curvatures=(-0.5, 0.1),
        width_angle=np.deg2rad(30.0),
        curvature_angle=np.deg2rad(-50.0),
    ),
}


def Psi_w_of(name):
    return astigmatic_gaussian_Psi_w(K0, **BEAMS[name])


def frame_from_definition(K_launch_cartesian):
    g_hat = K_launch_cartesian / np.linalg.norm(K_launch_cartesian)
    Z_hat = np.array([0.0, 0.0, 1.0])
    y_hat = Z_hat - (Z_hat @ g_hat) * g_hat
    y_hat /= np.linalg.norm(y_hat)
    x_hat = np.cross(y_hat, g_hat)
    return x_hat, y_hat, g_hat


def project(Psi_3D, x_hat, y_hat, g_hat):
    """Psi_w (x, y block) and the largest |component| involving g_hat"""
    B = np.column_stack([x_hat, y_hat, g_hat])
    P = B.T @ Psi_3D @ B
    return P[:2, :2], np.abs(P[2, :]).max()


def vacuum_Psi_w(Psi_w0, s):
    return np.linalg.inv(np.linalg.inv(Psi_w0) + s / K0 * np.eye(2))


def run(tmp_path, **kwargs):
    parameters = get_parameters_for_Scotty("DBS_synthetic")
    parameters.update(console_log_level="WARNING", output_path=tmp_path, len_tau=52)
    parameters.update(kwargs)
    return beam_me_up(**parameters)


def astig_kwargs(name):
    return dict(
        launch_beam_width=None,
        launch_beam_curvature=None,
        launch_beam_Psi_w=Psi_w_of(name),
    )


# ---------------------------------------------------------------- 1. the helper


@pytest.mark.parametrize("name", BEAMS)
def test_helper_principal_widths(name):
    beam = BEAMS[name]
    Psi_w = Psi_w_of(name)
    assert_allclose(Psi_w, Psi_w.T, atol=0)
    widths, axes = principal_widths(Psi_w)
    assert_allclose(widths, sorted(beam["widths"]), rtol=1e-13)
    angle = beam.get("width_angle", 0.0)
    e1 = np.array([np.cos(angle), np.sin(angle)])  # along widths[0] = 0.03, the smaller
    assert_allclose(abs(axes[:, 0] @ e1), 1.0, rtol=1e-13)
    curvatures = np.linalg.eigvalsh(np.real(Psi_w)) / K0
    assert_allclose(curvatures, sorted(beam["curvatures"]), rtol=1e-12)


def test_helper_circular_limit():
    for angle in (0.0, 0.7):
        assert_allclose(
            astigmatic_gaussian_Psi_w(K0, (0.04, 0.04), (-0.25, -0.25), angle, -angle),
            circular_gaussian_Psi_w(K0, 0.04, -0.25),
            rtol=0,
            atol=1e-14 * abs(circular_gaussian_Psi_w(K0, 0.04, -0.25)[0, 0]),
        )


# ------------------------------------------------- 2, 3. launch and analytic vacuum


@pytest.fixture(scope="module", params=list(BEAMS))
def analytic_run(request, tmp_path_factory):
    dt = run(
        tmp_path_factory.mktemp("astig_analytic"),
        launch_flag="vacuum_analytic",
        Psi_BC_flag="continuous",
        rtol=1e-8,
        atol=1e-10,
        **astig_kwargs(request.param),
    )
    return request.param, dt


def test_launch_Psi_is_the_given_one(analytic_run):
    name, dt = analytic_run
    so = dt["solver"]
    frame = frame_from_definition(so["K_launch_cartesian"].values)
    Psi_w, g_part = project(so["Psi_3D_launch_labframe_cartesian"].values, *frame)
    scale = np.abs(Psi_w_of(name)).max()
    assert_allclose(Psi_w, Psi_w_of(name), rtol=0, atol=1e-12 * scale)
    assert g_part < 1e-12 * scale
    assert_allclose(dt["inputs"]["launch_beam_Psi_w"].values, Psi_w_of(name), atol=0)


def test_entry_Psi_is_the_analytic_propagation(analytic_run):
    name, dt = analytic_run
    so = dt["solver"]
    d = float(so["distance_from_launch_to_entry"])
    assert d > 0.1  # a real vacuum gap (DBS_synthetic: about 0.4 m)
    frame = frame_from_definition(so["K_launch_cartesian"].values)
    Psi_w, g_part = project(so["Psi_3D_entry_labframe_cartesian"].values, *frame)
    expected = vacuum_Psi_w(Psi_w_of(name), d)
    scale = np.abs(expected).max()
    assert_allclose(Psi_w, expected, rtol=0, atol=1e-11 * scale)
    assert g_part < 1e-11 * scale
    # the beam has changed over the gap (the test is not trivially passed)
    assert np.abs(expected - Psi_w_of(name)).max() > 0.05 * scale


# ----------------------------------------------- 4, 5. numerical vacuum propagation


@pytest.fixture(scope="module", params=list(BEAMS))
def numerical_run(request, tmp_path_factory):
    dt = run(
        tmp_path_factory.mktemp("astig_numerical"),
        launch_flag="vacuum_numerical",
        Psi_BC_flag=None,
        rtol=1e-8,
        atol=1e-10,
        # finer than the analytic run, so that interpolating it is a small part of the difference
        len_tau=402,
        **astig_kwargs(request.param),
    )
    return request.param, dt


def _vacuum_points(dt):
    """Output points in vacuum (straight ray, K unchanged), excluding the last before the boundary"""
    so = dt["solver"]
    K = so["K_output_cartesian"].values
    K_launch = so["K_launch_cartesian"].values
    in_vacuum = np.linalg.norm(K - K_launch, axis=1) < 1e-9 * np.linalg.norm(K_launch)
    in_vacuum &= dt["beam"]["polflux"].values > 1.0 + 1e-6
    idx = np.flatnonzero(in_vacuum)
    # contiguous from the antenna
    idx = idx[: np.argmax(np.diff(np.r_[idx, -10]) != 1) + 1]
    return idx


def test_numerical_vacuum_matches_analytic(numerical_run):
    name, dt = numerical_run
    so = dt["solver"]
    idx = _vacuum_points(dt)
    assert len(idx) >= 10, len(idx)
    q = so["q_output_cartesian"].values
    q_launch = q[0]
    frame = frame_from_definition(so["K_launch_cartesian"].values)
    Psi_3D = so["Psi_3D_output_labframe_cartesian"].values
    worst = 0.0
    for i in idx:
        s = np.linalg.norm(q[i] - q_launch)
        expected = vacuum_Psi_w(Psi_w_of(name), s)
        Psi_w, g_part = project(Psi_3D[i], *frame)
        scale = np.abs(expected).max()
        worst = max(worst, np.abs(Psi_w - expected).max() / scale, g_part / scale)
    s_last = np.linalg.norm(q[idx[-1]] - q_launch)
    assert s_last > 0.1
    assert worst < 1e-6, worst


def test_continuous_BC_agrees_with_numerical_in_the_plasma(analytic_run, numerical_run):
    """Both integrate the same beam into the plasma; the analytic one restarts at the entry
    with the continuous BC. Compared at matched positions along the ray in the plasma"""
    name_a, dt_a = analytic_run
    name_n, dt_n = numerical_run
    if name_a != name_n:
        pytest.skip("pairs only")
    so_a, so_n = dt_a["solver"], dt_n["solver"]
    tau_a = so_a["tau"].values
    tau_n = so_n["tau"].values
    # tau offset: where the numerical ray reaches the analytic run's entry point
    q_entry = so_a["q_output_cartesian"].values[0]
    q_n = so_n["q_output_cartesian"].values
    d_n = np.linalg.norm(q_n - q_n[0], axis=1)
    d_entry = np.linalg.norm(q_entry - q_n[0])
    i = np.searchsorted(d_n, d_entry)
    # vacuum is linear in tau, so interpolate the offset there
    tau_entry = np.interp(d_entry, d_n[: i + 1], tau_n[: i + 1])
    Psi_a = so_a["Psi_3D_output_labframe_cartesian"].values
    Psi_n = so_n["Psi_3D_output_labframe_cartesian"].values

    def rel_difference(k):
        t = tau_a[k] + tau_entry
        Psi_n_at = np.array(
            [
                [
                    np.interp(t, tau_n, Psi_n[:, r, c].real)
                    + 1j * np.interp(t, tau_n, Psi_n[:, r, c].imag)
                    for c in range(3)
                ]
                for r in range(3)
            ]
        )
        return np.abs(Psi_n_at - Psi_a[k]).max() / np.abs(Psi_a[k]).max()

    # The continuous BC changes Psi a lot at this density's kink (psi = 1), so the BC is really tested:
    Psi_entry = so_a["Psi_3D_entry_labframe_cartesian"].values
    Psi_initial = so_a["Psi_3D_initial_labframe_cartesian"].values
    jump = np.abs(Psi_initial - Psi_entry).max() / np.abs(Psi_entry).max()
    assert jump > 1, jump
    # Right after the entry (the first output point; the entry itself is excluded, since there the
    # numerical run's output straddles the kink and interpolating it lands mid-jump)
    just_after = rel_difference(1)
    assert just_after < 1e-4, just_after

    # inside the plasma, up to half way to the end (well before the cut-off's steep Psi)
    picks = np.flatnonzero((tau_a > 0.05 * tau_a[-1]) & (tau_a < 0.5 * tau_a[-1]))
    worst = max(rel_difference(k) for k in picks)
    assert len(picks) >= 5
    # tolerance: the two routes differ by a floor of ~4e-4 that grows along the ray, the same for a
    # circular beam, and does not shrink with rtol or len_tau (astig_bc_convergence.py,
    # astig_bc_floor.py): it predates the astigmatic change. 1e-3 sits above it
    assert worst < 1e-3, worst


# ------------------------------------------------------ 6. circular via Psi_w


def test_circular_as_Psi_w_is_identical(tmp_path):
    common = dict(launch_flag="vacuum_analytic", Psi_BC_flag="continuous", len_tau=22)
    p = get_parameters_for_Scotty("DBS_synthetic")
    W, C = p["launch_beam_width"], p["launch_beam_curvature"]
    dt1 = run(tmp_path / "a", **common)
    dt2 = run(
        tmp_path / "b",
        launch_beam_width=None,
        launch_beam_curvature=None,
        launch_beam_Psi_w=circular_gaussian_Psi_w(K0, W, C),
        **common,
    )
    for key in (
        "Psi_3D_output_labframe_cartesian",
        "q_output_cartesian",
        "K_output_cartesian",
    ):
        assert_allclose(
            dt2["solver"][key].values, dt1["solver"][key].values, rtol=0, atol=0
        )
    assert_allclose(
        float(dt2["beam"]["beam_waist"]), float(dt1["beam"]["beam_waist"]), rtol=1e-12
    )


def test_rejects_both_and_arrays(tmp_path):
    with pytest.raises(ValueError, match="not both"):
        run(
            tmp_path,
            launch_flag="vacuum_analytic",
            Psi_BC_flag="continuous",
            launch_beam_Psi_w=Psi_w_of("elliptical"),
        )
    with pytest.raises(ValueError, match="scalar"):
        run(
            tmp_path,
            launch_flag="vacuum_analytic",
            Psi_BC_flag="continuous",
            launch_beam_width=[0.03, 0.05],
        )
    with pytest.raises(ValueError, match="positive definite"):
        run(
            tmp_path,
            launch_flag="vacuum_analytic",
            Psi_BC_flag="continuous",
            launch_beam_width=None,
            launch_beam_curvature=None,
            launch_beam_Psi_w=np.array([[1 + 2j, 0], [0, 1 - 2j]]),
        )
    with pytest.raises(ValueError, match="finite"):
        run(
            tmp_path,
            launch_flag="vacuum_analytic",
            Psi_BC_flag="continuous",
            launch_beam_width=None,
            launch_beam_curvature=None,
            launch_beam_Psi_w=np.full((2, 2), np.nan),
        )
    with pytest.raises(ValueError, match="launch_flag"):
        run(
            tmp_path,
            launch_flag="Plasma",
            Psi_BC_flag="continuous",
            launch_beam_width=None,
            launch_beam_curvature=None,
        )


def test_frame_upward_and_right_handed():
    """Scotty's frame against the definition, including past the vertical; a vertical launch is
    refused for a non-circular beam only"""
    from scotty.launch import launch_beam_frame, launch_beam_frame_is_defined

    for p, t in [(0, 0), (6, 0), (-20, 15), (60, -30), (120, 0), (-135, 10)]:
        x, y, g = launch_beam_frame(p, t)
        x_d, y_d, g_d = frame_from_definition(g)
        assert_allclose(np.c_[x, y, g], np.c_[x_d, y_d, g_d], atol=1e-14)
        assert y[2] > 0
    assert not launch_beam_frame_is_defined(90.0) and launch_beam_frame_is_defined(89.0)


def test_vertical_launch(tmp_path):
    common = dict(
        launch_flag="vacuum_analytic",
        Psi_BC_flag="continuous",
        ray_tracing_flag=False,
        poloidal_launch_angle_Torbeam=90.0,
        len_tau=12,
    )
    with pytest.raises(ValueError, match="vertical launch"):
        run(
            tmp_path,
            launch_beam_width=None,
            launch_beam_curvature=None,
            launch_beam_Psi_w=Psi_w_of("elliptical"),
            **common,
        )
