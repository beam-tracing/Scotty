r"""Scotty against the analytic Gaussian beam in a 2D linear layer (O-mode).

The analytic solution is from J. Ruiz Ruiz, F. I. Parra, V. H. Hall-Chen et
al., "Beam focusing and consequences for Doppler Backscattering
measurements", under consideration for J. Plasma Phys. (section 2.3, eqs.
(2.5), (2.6); Appendix A, eqs. (A2), (A4), (A10), (A11)). It is written here
in Scotty's unnormalised variables: Scotty's :math:`\tau` and :math:`\Psi`
rather than the paper's :math:`\tau' = \tau / (K_0 L)` and
:math:`\Psi' = \Psi L / K_0`.

Setup, in Cartesian lab coordinates :math:`(X, Y, Z)`: a uniform field
:math:`\mathbf{B} = -B_0 \hat{\mathbf{z}}`, and a density
:math:`\omega_{pe}^2 / \Omega^2 = X / L` for :math:`X > 0` (vacuum for
:math:`X < 0`). The field is an `InterpolatedField_Cartesian` on a grid, with
poloidal flux :math:`\psi = 1 - X / L`, and the density is
``LinearFit(1.0, n_c)`` with :math:`n_c` the O-mode cut-off density. The
beam is launched from inside the plasma (``launch_flag="plasma"``), just
after the edge, at the analytic :math:`q`, :math:`K` and :math:`\Psi`.

For the O-mode with :math:`K \perp B`, :math:`N^2 = 1 - X/L` and Scotty's
:math:`H = K^2 / K_0^2 - N^2`, so in the plane of the ray
:math:`dq/d\tau = 2 K / K_0^2`, :math:`dK/d\tau = -\hat{x} / L` and

.. math::
    \Psi^{-1}(\tau) = \Psi^{-1}(0) + \frac{2 \tau}{K_0^2},

with :math:`\Psi(0)` at the edge from (A4) and the beam-frame width and
curvature from (A9)-(A11). This is the paper's solution, and it holds for the
in-plane 2x2 block of :math:`\Psi`. On the ray :math:`K_z = 0`, and by symmetry
:math:`\partial^2 H / \partial K_z \partial K_{x,y} = 0`, so :math:`\Psi_{xz}`
and :math:`\Psi_{yz}` stay zero and :math:`\Psi_{zz}` does not affect the
plane.

Beyond the paper, :math:`\Psi_{zz}` (along :math:`B`). The paper's (2.4),
:math:`H = K^2/K_0^2 - 1 + \omega_{pe}^2/\Omega^2`, is the magnetised O-mode
for :math:`K \perp B`, which is all its 2D problem (no :math:`z` dependence)
needs; the paper makes no prediction for :math:`\Psi_{zz}`. Scotty's 3D beam
still needs a launch value for it, and Scotty evolves it with
:math:`\partial^2 H / \partial K_z^2`, the curvature of :math:`H` as
:math:`K` tilts towards :math:`B`. Expanding the cold-plasma (Appleton-Hartree)
O-mode near :math:`K \perp B` gives
:math:`N^2 \approx 1 - X + X (1 - X) K_\parallel^2 / K_\perp^2` for any
:math:`B_0 \neq 0` (for :math:`|K_\parallel| / K \lesssim Y / 2(1 - X)`), so
on the ray :math:`\partial^2 H / \partial K_z^2 = 2 (1 - X) / K_0^2` and

.. math::
    \Psi_{zz}^{-1}(\tau) = \Psi_{zz}^{-1}(0)
    + \frac{2}{K_0^2} \int_0^\tau (1 - X) \, d\tau.

`test_Psi_zz_magnetised_curvature` checks Scotty against this. It is a check
of Scotty's :math:`K_\parallel` dependence of the O-mode, not of the paper.
Taking (2.4) for every direction of :math:`K` instead (an isotropic model,
exact only at :math:`B = 0`) would give :math:`2 / K_0^2`, and a
:math:`\Psi_{zz}` up to 15% of :math:`\max|\Psi_{zz}|` different from Scotty's
here (at 10°).

Numerical choices:

- The spatial finite-difference steps are 1e-3 m rather than the default
  1e-5 m. :math:`H` is linear in :math:`q` here, so the stencils have no
  truncation error at any step, and the larger step keeps the rounding noise
  in :math:`\partial^2 H / \partial q^2` (about :math:`10^{-16} / \delta^2`)
  well below ``atol`` for the entries of :math:`\Psi` that are zero. With
  1e-5 m that noise is about 1e-6 m⁻², and with ``atol`` = 1e-10 the beam
  solver takes steps of about 1e-4 in :math:`\tau`.
- The steps point into the plasma (``auto_delta_sign``), so the one-sided
  stencils never sample the vacuum, where the density has its kink. With
  the X step pointing out of the plasma instead, the launch point (~1e-6 m
  inside the edge) samples the vacuum and Psi is wrong by ~100%.
- :math:`\omega_{ce} / \Omega = 0.3`, except at 10°, which uses 0.1. Scotty's
  O-mode root, :math:`(\beta - \sqrt{\beta^2 - 4\alpha\gamma}) / 2\alpha`, is
  0/0 where Booker's :math:`\alpha = \epsilon_\perp = 0` (the X-mode's upper
  hybrid layer, :math:`X = 1 - Y^2`). At 10° the O-mode reaches
  :math:`X = 0.97` before its cut-off, beyond :math:`1 - 0.3^2 = 0.91`, and
  the beam solver fails there; with 0.1 the layer is at 0.99. Values where
  the ratio is exactly 0.5 or 1 also stop the ray at the start (the
  resonance event is zero everywhere in a uniform field).
"""

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy import constants
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

from scotty.beam_me_up import beam_me_up
from scotty.geometry import InterpolatedField_Cartesian
from scotty.profile_fit import LinearFit

##################################################
# Parameters (JET 97080, as in the paper): K0 L = 1600, K0 W0 = 22
##################################################

LAUNCH_FREQ_GHZ = 2200.0 * constants.c / (2 * np.pi) / 1e9  # K0 = 2200 m^-1
OMEGA = 2 * np.pi * LAUNCH_FREQ_GHZ * 1e9
K0 = OMEGA / constants.c
L = 1600 / K0
W0 = 22 / K0
# O-mode cut-off density at OMEGA, in 10^19 m^-3
N_CUTOFF = constants.epsilon_0 * constants.m_e * OMEGA**2 / constants.e**2 / 1e19

# Scotty's tau where the beam is launched, counted from the plasma edge
TAU_LAUNCH = 1e-3
LEN_TAU = 200
# rtol = 1e-8 takes ~20 min per run, as the rounding noise in d^2H/dq^2 then
# limits the steps; at 1e-7 a run takes ~45 s
SOLVER = {"rtol": 1e-7, "atol": 1e-8}
DELTA_Q = 1e-3

# Tolerances, relative to L, K0, max |Psi|, the output step in tau, or the
# value itself. Largest error measured over the cases in brackets
TOL = {
    "q": 1e-11,  # (1.8e-12)
    "K": 2e-11,  # (3.8e-12)
    "Psi": 2e-6,  # in-plane block, real and imaginary parts (4.6e-7, at 10 deg)
    "Psi_off_plane": 3e-6,  # Psi_xz, Psi_yz, exactly zero (7.0e-7, at 10 deg)
    "Psi_zz": 1e-7,  # (1.7e-8)
    "W": 3e-6,  # (5.4e-7, at 10 deg)
    "focus": 3e-6,  # fraction of the output step (5.6e-7)
    "tau_leave": 1e-8,  # (2.0e-9)
    "tau_cutoff": 1e-7,  # (1.5e-8)
    "B0_tau": 1e-9,  # (1.6e-10)
    "B0_ray": 3e-9,  # q, K (4.7e-10)
    "B0_Psi": 1.5e-6,  # (2.9e-7)
}


def B0_for(gyro_ratio):
    """|B| for which omega_ce / Omega = `gyro_ratio`"""
    return gyro_ratio * OMEGA * constants.m_e / constants.e


##################################################
# The analytic solution
##################################################


def analytic_ray(tau, a0):
    """q and K, shape (N, 3), at Scotty's tau from the edge, for a ray that
    enters at (0, 0, 0) with K = K0 (cos a0, sin a0, 0)"""
    c, s = np.cos(a0), np.sin(a0)
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    K_x = K0 * c - tau / L
    zero = np.zeros_like(tau)
    q = np.stack([L * (c**2 - (K_x / K0) ** 2), 2 * s * tau / K0, zero], axis=-1)
    K = np.stack([K_x, np.full_like(tau, K0 * s), zero], axis=-1)
    return q, K


def edge_Psi_YY(R_Y):
    """Psi in the beam frame, perpendicular to K in the plane, at the edge"""
    return K0 / R_Y + 2j / W0**2


def edge_Psi(a0, R_Y):
    """Lab-frame Psi at the edge, (A4) in unnormalised form, with Psi_zz from
    the same width and curvature as Psi_YY (a circular beam)"""
    c, s = np.cos(a0), np.sin(a0)
    t = s / c
    # Normalised Psi' = Psi L / K0; (A11) gives Psi'_yy from Psi'_YY
    Psi_YY_n = edge_Psi_YY(R_Y) * L / K0
    Psi_yy_n = c**2 * (Psi_YY_n + s**2 / (2 * c))
    Psi_n = np.array(
        [
            [Psi_yy_n * t**2 - 1 / (2 * c), -Psi_yy_n * t, 0],
            [-Psi_yy_n * t, Psi_yy_n, 0],
            [0, 0, Psi_YY_n],
        ],
        dtype=complex,
    )
    return Psi_n * K0 / L


def analytic_Psi_in_plane(tau, a0, R_Y):
    """Lab-frame Psi from Psi^-1(tau) = Psi^-1(0) + 2 tau / K0^2, shape (N, 3, 3).
    Its zz entry is the isotropic 2 / K0^2 model, not Scotty's (see
    `analytic_Psi_zz`)"""
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    inverse = np.linalg.inv(edge_Psi(a0, R_Y))
    return np.linalg.inv(
        inverse[None] + (2 * tau / K0**2)[:, None, None] * np.eye(3)[None]
    )


def analytic_Psi_zz(tau, a0, R_Y):
    """Psi_zz for the magnetised O-mode: d^2 H / dK_z^2 = 2 (1 - X) / K0^2"""
    c, s = np.cos(a0), np.sin(a0)
    tau = np.asarray(tau, dtype=float)
    integral = tau * s**2 + K0 * L * (c**3 - (c - tau / (K0 * L)) ** 3) / 3
    return 1 / (1 / edge_Psi_YY(R_Y) + 2 * integral / K0**2)


def Psi_YY_from_lab(Psi, K):
    """(A10): Psi in the plane, perpendicular to K"""
    K_magnitude = np.linalg.norm(K[:, :2], axis=1)
    cos_a, sin_a = K[:, 0] / K_magnitude, K[:, 1] / K_magnitude
    return (
        Psi[:, 0, 0] * sin_a**2
        - 2 * Psi[:, 0, 1] * sin_a * cos_a
        + Psi[:, 1, 1] * cos_a**2
    )


def Psi_YY_closed_form(tau, a0, R_Y):
    """Psi_YY as a function of K_x, from inverting the 2x2 in-plane block by
    hand (the content of (2.6)); an independent check of the matrix route"""
    c, s = np.cos(a0), np.sin(a0)
    Psi_yy_n = c**2 * (edge_Psi_YY(R_Y) * L / K0 + s**2 / (2 * c))
    tau_n = np.asarray(tau, dtype=float) / (K0 * L)
    K_x = c - tau_n  # normalised to K0
    d = 2 * tau_n - 2 * s**2 / c + 1 / Psi_yy_n
    numerator = d * s**2 - 4 * s**2 * K_x - 2 * K_x**3
    denominator = (K_x**2 + s**2) * (-2 * K_x * d - 4 * s**2)
    return numerator / denominator * K0 / L


def width(Psi_YY):
    return np.sqrt(2 / np.imag(Psi_YY))


def analytic_focus(a0, R_Y):
    """tau (from the edge) and width of the narrowest point of the beam in the
    plasma; None if the narrowest point is at the edge"""
    tau_leave = 2 * K0 * L * np.cos(a0)
    tau = np.linspace(0, tau_leave, 20001)
    W = width(Psi_YY_closed_form(tau, a0, R_Y))
    i = int(np.argmin(W))
    if i == 0:
        return None
    result = minimize_scalar(
        lambda t: width(Psi_YY_closed_form(t, a0, R_Y)),
        bounds=(tau[i - 1], tau[i + 1]),
        method="bounded",
        options={"xatol": 1e-9},
    )
    return result.x, result.fun


##################################################
# Scotty runs
##################################################

# (launch angle in degrees, R_Y = R_z at the edge, omega_ce / Omega)
CASES = {
    "a10": (10.0, np.inf, 0.1),
    "a30": (30.0, np.inf, 0.3),
    "a50": (50.0, np.inf, 0.3),
    # Converging, R_Y0 = -2 L cos(a0) / sin(a0)^2 as in the discussion of (A16)
    "a30-curved": (
        30.0,
        -2 * L * np.cos(np.deg2rad(30)) / np.sin(np.deg2rad(30)) ** 2,
        0.3,
    ),
    # The a30 case at another |B|, for the B0 independence of the O-mode
    "a30-B0": (30.0, np.inf, 0.1),
}


def linear_layer_field(B0, a0):
    """Uniform B = -B0 z_hat and psi = 1 - X / L on a grid covering the ray,
    with margins in X (including vacuum, X < 0), Y and Z"""
    X = np.linspace(-0.1 * L, 1.1 * L, 13)
    Y = np.linspace(-0.1 * L, 2 * L * np.sin(2 * a0) + 0.1 * L, 13)
    Z = np.linspace(-0.1 * L, 0.1 * L, 7)
    X_grid, _, _ = np.meshgrid(X, Y, Z, indexing="ij")
    zero = np.zeros_like(X_grid)
    return InterpolatedField_Cartesian(X, Y, Z, zero, zero, zero - B0, 1 - X_grid / L)


def run_linear_layer(a0_deg, R_Y, gyro_ratio, output_path):
    a0 = np.deg2rad(a0_deg)
    q, K = analytic_ray(TAU_LAUNCH, a0)
    Psi = analytic_Psi_in_plane(TAU_LAUNCH, a0, R_Y)[0]
    Psi[2, 2] = analytic_Psi_zz(TAU_LAUNCH, a0, R_Y)
    return beam_me_up(
        geometry="cartesian",
        poloidal_launch_angle_Torbeam=0.0,
        toroidal_launch_angle_Torbeam=0.0,
        launch_freq_GHz=LAUNCH_FREQ_GHZ,
        launch_beam_width=W0,
        launch_beam_curvature=0.0,
        launch_position=q[0],
        mode_flag="O",
        launch_flag="plasma",
        Psi_BC_flag=None,
        plasmaLaunch_K_cartesian=K[0],
        plasmaLaunch_Psi_3D_lab_cartesian=Psi,
        find_B_method=linear_layer_field(B0_for(gyro_ratio), a0),
        density_fit_method=LinearFit(1.0, N_CUTOFF),
        poloidal_flux_zero_density=1.0,
        resonance_flag=False,
        delta_X=DELTA_Q,
        delta_Y=DELTA_Q,
        delta_Z=DELTA_Q,
        len_tau=LEN_TAU,
        figure_flag=False,
        detailed_analysis_flag=False,
        output_path=output_path,
        console_log_level="WARNING",
        **SOLVER,
    )


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Each case is run once, when first needed"""
    cache = {}

    def get(name):
        if name not in cache:
            a0_deg, R_Y, gyro_ratio = CASES[name]
            dt = run_linear_layer(
                a0_deg, R_Y, gyro_ratio, tmp_path_factory.mktemp(name)
            )
            so = dt["solver_output"]
            tau = so["tau"].values
            a0 = np.deg2rad(a0_deg)
            tau_edge = tau + TAU_LAUNCH
            cache[name] = {
                "a0": a0,
                "R_Y": R_Y,
                "tau": tau,
                "tau_edge": tau_edge,
                "q": so["q_output_cartesian"].values,
                "K": so["K_output_cartesian"].values,
                "Psi": so["Psi_3D_output_labframe_cartesian"].values,
                "analytic_ray": analytic_ray(tau_edge, a0),
                "analytic_Psi": analytic_Psi_in_plane(tau_edge, a0, R_Y),
            }
        return cache[name]

    return get


MAIN_CASES = ["a10", "a30", "a50", "a30-curved"]

##################################################
# The analytic solution itself
##################################################


@pytest.mark.parametrize("name", MAIN_CASES)
def test_analytic_solution_is_consistent(name):
    """The helpers satisfy H = 0, the constraint (A3)
    grad_K H . Psi + grad H = 0, the beam-frame edge values, and agree with
    the closed form of Psi_YY"""
    a0_deg, R_Y, _ = CASES[name]
    a0 = np.deg2rad(a0_deg)
    tau_leave = 2 * K0 * L * np.cos(a0)
    tau = np.linspace(0, tau_leave, 101)
    q, K = analytic_ray(tau, a0)
    Psi = analytic_Psi_in_plane(tau, a0, R_Y)

    # H = K^2 / K0^2 - 1 + X / L = 0, and the ray leaves at x = 0
    assert_allclose(np.sum(K**2, axis=1) / K0**2 - 1 + q[:, 0] / L, 0, atol=1e-14)
    assert abs(q[-1, 0]) < 1e-14 * L

    # (A3): (2 K / K0^2) . Psi = -x_hat / L, at the edge and along the ray
    constraint = np.einsum("ti,tij->tj", 2 * K / K0**2, Psi)
    expected = np.broadcast_to([-1 / L, 0, 0], constraint.shape)
    assert_allclose(constraint, expected, atol=1e-12 / L)

    # (A10), (A11) at the edge: Psi_YY = K0 / R_Y + 2i / W0^2
    assert_allclose(Psi_YY_from_lab(Psi[:1], K[:1])[0], edge_Psi_YY(R_Y), rtol=1e-13)
    # The closed form of Psi_YY(K_x) and the matrix route agree
    assert_allclose(
        Psi_YY_closed_form(tau, a0, R_Y), Psi_YY_from_lab(Psi, K), rtol=1e-12
    )


@pytest.mark.parametrize("name", MAIN_CASES)
def test_analytic_Psi_zz_is_consistent(name):
    """Beyond the paper: the closed form of Psi_zz for the magnetised O-mode
    agrees with integrating dPsi_zz/dtau = -2 (1 - X) Psi_zz^2 / K0^2"""
    a0_deg, R_Y, _ = CASES[name]
    a0 = np.deg2rad(a0_deg)
    tau_leave = 2 * K0 * L * np.cos(a0)
    tau = np.linspace(0, tau_leave, 101)

    def rhs(t, y):
        Psi_zz = y[0] + 1j * y[1]
        X = analytic_ray(t, a0)[0][0, 0] / L
        d = -2 * (1 - X) * Psi_zz**2 / K0**2
        return [d.real, d.imag]

    Psi_zz_0 = edge_Psi_YY(R_Y)
    solution = solve_ivp(
        rhs,
        (0, tau_leave),
        [Psi_zz_0.real, Psi_zz_0.imag],
        t_eval=tau,
        rtol=1e-11,
        atol=1e-9,
    )
    assert_allclose(
        solution.y[0] + 1j * solution.y[1],
        analytic_Psi_zz(tau, a0, R_Y),
        rtol=1e-8,
    )


##################################################
# Scotty against the analytic solution
##################################################


@pytest.mark.parametrize("name", MAIN_CASES)
def test_ray(runs, name):
    """q and K along the ray"""
    run = runs(name)
    q, K = run["analytic_ray"]
    assert len(run["tau"]) == LEN_TAU  # the uniform points and the cut-off
    assert_allclose(run["q"], q, rtol=0, atol=TOL["q"] * L)
    assert_allclose(run["K"], K, rtol=0, atol=TOL["K"] * K0)


@pytest.mark.parametrize("name", MAIN_CASES)
def test_Psi_in_plane(runs, name):
    """The in-plane 2x2 block of the lab-frame Psi, real and imaginary parts,
    and the zero entries coupling the plane to z"""
    run = runs(name)
    Psi, expected = run["Psi"], run["analytic_Psi"]
    scale = np.abs(expected[:, :2, :2]).max()
    assert_allclose(
        Psi[:, :2, :2].real,
        expected[:, :2, :2].real,
        rtol=0,
        atol=TOL["Psi"] * scale,
    )
    assert_allclose(
        Psi[:, :2, :2].imag,
        expected[:, :2, :2].imag,
        rtol=0,
        atol=TOL["Psi"] * scale,
    )
    assert np.abs(Psi[:, :2, 2]).max() < TOL["Psi_off_plane"] * scale
    assert np.abs(Psi[:, 2, :2]).max() < TOL["Psi_off_plane"] * scale


@pytest.mark.parametrize("name", MAIN_CASES)
def test_Psi_zz_magnetised_curvature(runs, name):
    """Beyond the paper: Psi_zz, along B, with the magnetised O-mode's
    d^2 H / dK_z^2 = 2 (1 - X) / K0^2 (module notes)"""
    run = runs(name)
    expected = analytic_Psi_zz(run["tau_edge"], run["a0"], run["R_Y"])
    assert_allclose(
        run["Psi"][:, 2, 2],
        expected,
        rtol=0,
        atol=TOL["Psi_zz"] * np.abs(expected).max(),
    )


def parabola_minimum(tau, W, step):
    """tau of the vertex of the parabola through the narrowest point of the
    uniformly spaced W and its two neighbours"""
    i = int(np.argmin(W))
    W_before, W_min, W_after = W[i - 1 : i + 2]
    offset = 0.5 * (W_before - W_after) / (W_before - 2 * W_min + W_after)
    return tau[i] + offset * step


@pytest.mark.parametrize("name", MAIN_CASES)
def test_width_and_focus(runs, name):
    """W_Y = sqrt(2 / Im Psi_YY) along the ray, and where the beam is
    narrowest.

    Scotty's output is sampled every `step` in tau, so the focus is located
    from a parabola through the three narrowest output points. The parabola's
    own error (up to 0.05 of a step at 10 deg, where the focus is sharp) is
    the same for Scotty's widths and the analytic widths at the same points,
    so those two are compared; the narrowest output point is also checked to
    be the one nearest the exact analytic focus"""
    run = runs(name)
    _, K = run["analytic_ray"]
    W = width(Psi_YY_from_lab(run["Psi"], run["K"]))
    W_expected = width(Psi_YY_from_lab(run["analytic_Psi"], K))
    assert_allclose(W, W_expected, rtol=TOL["W"])

    focus = analytic_focus(run["a0"], run["R_Y"])
    # The uniform output points (the cut-off is added to them)
    step = run["tau"][1]
    uniform = np.isclose(run["tau"] / step, np.round(run["tau"] / step), atol=1e-6)
    tau = run["tau_edge"][uniform]
    W, W_expected = W[uniform], W_expected[uniform]
    i = int(np.argmin(W))
    if focus is None:
        # Narrowest at the edge: W only grows along the ray
        assert i == 0
        return
    assert i == int(np.argmin(np.abs(tau - focus[0])))
    tau_focus = parabola_minimum(tau, W, step)
    tau_focus_expected = parabola_minimum(tau, W_expected, step)
    assert abs(tau_focus - tau_focus_expected) < TOL["focus"] * step
    assert abs(tau_focus_expected - focus[0]) < 0.1 * step


@pytest.mark.parametrize("name", MAIN_CASES)
def test_tau_leave_and_cutoff(runs, name):
    """The ray solver's exit tau against 2 K0 L cos(a0), and the cut-off
    (the point added to the output, where |K| is smallest) against
    K0 L cos(a0)"""
    run = runs(name)
    tau, a0 = run["tau"], run["a0"]
    step = tau[1]
    uniform = np.isclose(tau / step, np.round(tau / step), atol=1e-6)
    assert np.count_nonzero(~uniform) == 1
    # The output grid is linspace(0, tau_leave, LEN_TAU - 1, endpoint=False)
    tau_leave = step * (LEN_TAU - 1) + TAU_LAUNCH
    assert_allclose(tau_leave, 2 * K0 * L * np.cos(a0), rtol=TOL["tau_leave"])
    tau_cutoff = tau[~uniform][0] + TAU_LAUNCH
    assert_allclose(tau_cutoff, K0 * L * np.cos(a0), rtol=TOL["tau_cutoff"])


def test_B0_independence(runs):
    """The O-mode beam with K perpendicular to B does not depend on |B|:
    omega_ce / Omega = 0.3 and 0.1"""
    a, b = runs("a30"), runs("a30-B0")
    assert_allclose(b["tau"], a["tau"], rtol=TOL["B0_tau"])
    assert_allclose(b["q"], a["q"], rtol=0, atol=TOL["B0_ray"] * L)
    assert_allclose(b["K"], a["K"], rtol=0, atol=TOL["B0_ray"] * K0)
    scale = np.abs(a["Psi"]).max()
    assert_allclose(b["Psi"], a["Psi"], rtol=0, atol=TOL["B0_Psi"] * scale)
