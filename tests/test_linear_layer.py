r"""Scotty against the analytic Gaussian beam in a 2D linear layer (O-mode).

The analytic solution is from J. Ruiz Ruiz, F. I. Parra, V. H. Hall-Chen et
al., "Beam focusing and consequences for Doppler Backscattering
measurements", J. Plasma Phys. 91, E60 (2025),
doi:10.1017/S0022377825000170 (section 2.3, eqs. (2.9), (2.10); Appendix A,
eqs. (A.2), (A.4), (A.10), (A.11)). It is written here in Scotty's
unnormalised variables: Scotty's :math:`\tau` and :math:`\Psi` rather than the
paper's :math:`\tau' = \tau / (K_0 L)` and :math:`\Psi' = \Psi L / K_0`.

Notation is Scotty's: the paper's :math:`(x, y, z)` and
:math:`(\hat{X}, \hat{Y}, \hat{g})` are Scotty's :math:`(X, Y, Z)` and
:math:`(\hat{x}, \hat{y}, \hat{g})`.

Setup, in Cartesian lab coordinates :math:`(X, Y, Z)`: a uniform field
:math:`\mathbf{B} = -B_0 \hat{\mathbf{Z}}`, and a density
:math:`\Omega_{pe}^2 / \Omega^2 = X / L` for :math:`X > 0` (vacuum for
:math:`X < 0`). The field is an `InterpolatedField_Cartesian` on a grid, with
poloidal flux :math:`\psi = 1 - X / L`, and the density is
``LinearFit(1.0, n_c)`` with :math:`n_c` the O-mode cut-off density. The
beam is launched from inside the plasma (``launch_flag="plasma"``), just
after the edge, at the analytic :math:`q`, :math:`K` and :math:`\Psi`.

For the O-mode with :math:`K \perp B`, :math:`N^2 = 1 - X/L` and Scotty's
:math:`H = K^2 / K_0^2 - N^2`, so in the plane of the ray
:math:`dq/d\tau = 2 K / K_0^2`, :math:`dK/d\tau = -\hat{X} / L` and

.. math::
    \Psi^{-1}(\tau) = \Psi^{-1}(0) + \frac{2 \tau}{K_0^2},

with :math:`\Psi(0)` at the edge from (A.4) and the beam-frame width and
curvature from (A.9)-(A.11). This is the paper's solution, and it holds for the
in-plane 2x2 block of :math:`\Psi`. On the ray :math:`K_Z = 0`, and by symmetry
:math:`\partial^2 H / \partial K_Z \partial K_{X,Y} = 0`, so :math:`\Psi_{XZ}`
and :math:`\Psi_{YZ}` stay zero and :math:`\Psi_{ZZ}` does not affect the
plane.

A correction to the paper: :math:`\Psi_{ZZ}`, along :math:`B`. Appendix A of
the paper takes :math:`\nabla_K \nabla_K H = (2 / K_0^2) I`, and its section
2.3 says that the vacuum solution is also the solution for the width along
:math:`B` (:math:`W_x` here). That is (2.8),
:math:`H = K^2/K_0^2 - 1 + \Omega_{pe}^2/\Omega^2`, taken for every direction
of :math:`K`, which is exact only at :math:`B = 0`: for a magnetised plasma
(2.8) is the O-mode only for :math:`K \perp B`. Scotty's 3D beam evolves
:math:`\Psi_{ZZ}` with :math:`\partial^2 H / \partial K_Z^2`, the curvature of
:math:`H` as :math:`K` tilts towards :math:`B`. Expanding the cold-plasma
(Appleton-Hartree) O-mode near :math:`K \perp B` gives

.. math::
    N^2 \approx 1 - \frac{\Omega_{pe}^2}{\Omega^2}
    + \frac{\Omega_{pe}^2}{\Omega^2} \left(1 - \frac{\Omega_{pe}^2}{\Omega^2}\right)
    \frac{K_\parallel^2}{K_\perp^2}

for any :math:`B_0 \neq 0` (for :math:`|K_\parallel| / K \lesssim
(\Omega_{ce} / \Omega) / 2 (1 - \Omega_{pe}^2 / \Omega^2)`), so on the ray
:math:`\partial^2 H / \partial K_Z^2 = 2 (1 - \Omega_{pe}^2 / \Omega^2) / K_0^2`
and

.. math::
    \Psi_{ZZ}^{-1}(\tau) = \Psi_{ZZ}^{-1}(0)
    + \frac{2}{K_0^2} \int_0^\tau \left(1 - \frac{\Omega_{pe}^2}{\Omega^2}\right) d\tau.

This differs from the paper's :math:`2 / K_0^2` for a magnetised plasma, so
`test_Psi_ZZ_magnetised_curvature` is a correction to that statement: it
checks Scotty against the magnetised form, not against the paper. The paper's
form would give a :math:`\Psi_{ZZ}` up to 15% of :math:`\max|\Psi_{ZZ}|`
different from Scotty's here (at 10°).

What this test cannot catch: :math:`\nabla \nabla H` and
:math:`\nabla_K \nabla H` vanish in this slab, so the beam equation is tested
only through its :math:`\nabla_K \nabla_K H` term, and an error in the other
terms would not show.

Numerical choices:

- The spatial finite-difference steps are 1e-3 m rather than the default
  1e-5 m. :math:`H` is linear in :math:`q` here, so the stencils have no
  truncation error at any step, and the larger step keeps the rounding noise
  in :math:`\partial^2 H / \partial q^2` (about :math:`10^{-16} / \delta^2`)
  well below the beam solver's absolute tolerance for :math:`\Psi`, about
  2e-7 m⁻² here. With 1e-5 m that noise is about 1e-6 m⁻², above it.
- The steps point into the plasma (``auto_delta_sign``), so the one-sided
  stencils never sample the vacuum, where the density has its kink. With
  the X step pointing out of the plasma instead, the launch point (~1e-6 m
  inside the edge) samples the vacuum and Psi is wrong by ~100%.
- :math:`\Omega_{ce} / \Omega = 0.3`. A ratio of exactly 0.5 or 1 in this
  uniform field would put the whole ray on a resonance.
- Run time: nearly all of it is the beam solve, about 16 ms per evaluation and
  about 1100-1600 evaluations per run.
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
# The errors don't fall smoothly with these, as they depend on where the steps
# land: at rtol = 1e-7, atol = 3e-11 misses the Psi_ZZ limit by 1%. At this atol,
# rtol = 1e-7 and 1e-8 also pass, with errors up to 0.48 of the limits
SOLVER = {"rtol": 3e-8, "atol": 1e-11}
DELTA_Q = 1e-3

# Tolerances, relative to L, K0, max |Psi|, the output step in tau, or the
# value itself. Largest error measured over the cases in brackets
TOL = {
    "q": 1e-11,  # (1.4e-12)
    "K": 2e-11,  # (2.0e-13)
    # The in-plane block, real and imaginary parts. The error moves between
    # machines (8.0e-7, at 10 deg)
    "Psi": 5e-6,
    "Psi_off_plane": 3e-6,  # Psi_XZ, Psi_YZ, exactly zero (7.0e-7, at 10 deg)
    "Psi_ZZ": 1e-7,  # (3.1e-8)
    "W": 3e-6,  # (8.2e-7, at 10 deg)
    "focus": 3e-6,  # fraction of the output step (7.0e-7)
    "tau_leave": 1e-8,  # (3.0e-11)
    "tau_cutoff": 1e-7,  # (4.7e-9)
    "B0_tau": 1e-9,  # (3.4e-11)
    "B0_ray": 3e-9,  # q, K (1.0e-10)
    "B0_Psi": 1.5e-6,  # (1.4e-7)
}


def B0_for(gyro_ratio):
    """|B| for which Omega_ce / Omega = `gyro_ratio`"""
    return gyro_ratio * OMEGA * constants.m_e / constants.e


##################################################
# The analytic solution
##################################################


def analytic_ray(tau, a0):
    """q and K, shape (N, 3), at Scotty's tau from the edge, for a ray that
    enters at (0, 0, 0) with K = K0 (cos a0, sin a0, 0)"""
    c, s = np.cos(a0), np.sin(a0)
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    K_X = K0 * c - tau / L
    zero = np.zeros_like(tau)
    q = np.stack([L * (c**2 - (K_X / K0) ** 2), 2 * s * tau / K0, zero], axis=-1)
    K = np.stack([K_X, np.full_like(tau, K0 * s), zero], axis=-1)
    return q, K


def edge_Psi_yy(R_y):
    """Psi in the beam frame, along y_hat (perpendicular to K in the plane), at
    the edge"""
    return K0 / R_y + 2j / W0**2


def edge_Psi(a0, R_y):
    """Lab-frame Psi at the edge, (A.4) in unnormalised form, with Psi_ZZ from
    the same width and curvature as the beam-frame Psi_yy (a circular beam)"""
    c, s = np.cos(a0), np.sin(a0)
    t = s / c
    # Normalised Psi' = Psi L / K0; (A.11) gives the lab Psi'_YY from the beam
    # Psi'_yy
    Psi_yy_n = edge_Psi_yy(R_y) * L / K0
    Psi_YY_n = c**2 * (Psi_yy_n + s**2 / (2 * c))
    Psi_n = np.array(
        [
            [Psi_YY_n * t**2 - 1 / (2 * c), -Psi_YY_n * t, 0],
            [-Psi_YY_n * t, Psi_YY_n, 0],
            [0, 0, Psi_yy_n],
        ],
        dtype=complex,
    )
    return Psi_n * K0 / L


def analytic_Psi_in_plane(tau, a0, R_y):
    """Lab-frame Psi from Psi^-1(tau) = Psi^-1(0) + 2 tau / K0^2, shape (N, 3, 3).
    Its ZZ entry is the paper's 2 / K0^2, not Scotty's (see `analytic_Psi_ZZ`)"""
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    inverse = np.linalg.inv(edge_Psi(a0, R_y))
    return np.linalg.inv(
        inverse[None] + (2 * tau / K0**2)[:, None, None] * np.eye(3)[None]
    )


def analytic_Psi_ZZ(tau, a0, R_y):
    """Psi_ZZ for the magnetised O-mode:
    d^2 H / dK_Z^2 = 2 (1 - Omega_pe^2 / Omega^2) / K0^2"""
    c, s = np.cos(a0), np.sin(a0)
    tau = np.asarray(tau, dtype=float)
    integral = tau * s**2 + K0 * L * (c**3 - (c - tau / (K0 * L)) ** 3) / 3
    return 1 / (1 / edge_Psi_yy(R_y) + 2 * integral / K0**2)


def Psi_yy_from_lab(Psi, K):
    """(A.11): Psi in the beam frame, along y_hat, from the lab frame"""
    K_magnitude = np.linalg.norm(K[:, :2], axis=1)
    cos_a, sin_a = K[:, 0] / K_magnitude, K[:, 1] / K_magnitude
    return (
        Psi[:, 0, 0] * sin_a**2
        - 2 * Psi[:, 0, 1] * sin_a * cos_a
        + Psi[:, 1, 1] * cos_a**2
    )


def Psi_yy_closed_form(tau, a0, R_y):
    """The beam-frame Psi_yy as a function of K_X, from inverting the 2x2 in-plane
    block by hand (the content of (2.10)); an independent check of the matrix
    route"""
    c, s = np.cos(a0), np.sin(a0)
    Psi_YY_n = c**2 * (edge_Psi_yy(R_y) * L / K0 + s**2 / (2 * c))
    tau_n = np.asarray(tau, dtype=float) / (K0 * L)
    K_X = c - tau_n  # normalised to K0
    d = 2 * tau_n - 2 * s**2 / c + 1 / Psi_YY_n
    numerator = d * s**2 - 4 * s**2 * K_X - 2 * K_X**3
    denominator = (K_X**2 + s**2) * (-2 * K_X * d - 4 * s**2)
    return numerator / denominator * K0 / L


def width(Psi_yy):
    return np.sqrt(2 / np.imag(Psi_yy))


def analytic_focus(a0, R_y):
    """tau (from the edge) and width of the narrowest point of the beam in the
    plasma; None if the narrowest point is at the edge"""
    tau_leave = 2 * K0 * L * np.cos(a0)
    tau = np.linspace(0, tau_leave, 20001)
    W = width(Psi_yy_closed_form(tau, a0, R_y))
    i = int(np.argmin(W))
    if i == 0:
        return None
    result = minimize_scalar(
        lambda t: width(Psi_yy_closed_form(t, a0, R_y)),
        bounds=(tau[i - 1], tau[i + 1]),
        method="bounded",
        options={"xatol": 1e-9},
    )
    return result.x, result.fun


##################################################
# Scotty runs
##################################################

# Converging, R_y0 = -2 L cos(a0) / sin(a0)^2 as in the discussion of (A.16)
R_Y_CURVED = -2 * L * np.cos(np.deg2rad(30)) / np.sin(np.deg2rad(30)) ** 2

# (launch angle in degrees, R_y = R_x at the edge, Omega_ce / Omega)
CASES = {
    "a10": (10.0, np.inf, 0.3),
    "a30-curved": (30.0, R_Y_CURVED, 0.3),
    # The a30-curved case at another |B|, for the B0 independence of the O-mode
    "a30-curved-B0": (30.0, R_Y_CURVED, 0.1),
}


def linear_layer_field(B0, a0):
    """Uniform B = -B0 Z_hat and psi = 1 - X / L on a grid covering the ray,
    with margins in X (including vacuum, X < 0), Y and Z"""
    X = np.linspace(-0.1 * L, 1.1 * L, 13)
    Y = np.linspace(-0.1 * L, 2 * L * np.sin(2 * a0) + 0.1 * L, 13)
    Z = np.linspace(-0.1 * L, 0.1 * L, 7)
    X_grid, _, _ = np.meshgrid(X, Y, Z, indexing="ij")
    zero = np.zeros_like(X_grid)
    return InterpolatedField_Cartesian(X, Y, Z, zero, zero, zero - B0, 1 - X_grid / L)


def run_linear_layer(a0_deg, R_y, gyro_ratio, output_path):
    a0 = np.deg2rad(a0_deg)
    q, K = analytic_ray(TAU_LAUNCH, a0)
    Psi = analytic_Psi_in_plane(TAU_LAUNCH, a0, R_y)[0]
    Psi[2, 2] = analytic_Psi_ZZ(TAU_LAUNCH, a0, R_y)
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
            a0_deg, R_y, gyro_ratio = CASES[name]
            dt = run_linear_layer(
                a0_deg, R_y, gyro_ratio, tmp_path_factory.mktemp(name)
            )
            so = dt["solver"]
            tau = so["tau"].values
            a0 = np.deg2rad(a0_deg)
            tau_edge = tau + TAU_LAUNCH
            cache[name] = {
                "a0": a0,
                "R_y": R_y,
                "tau": tau,
                "tau_edge": tau_edge,
                "q": so["q_output_cartesian"].values,
                "K": so["K_output_cartesian"].values,
                "Psi": so["Psi_3D_output_labframe_cartesian"].values,
                "analytic_ray": analytic_ray(tau_edge, a0),
                "analytic_Psi": analytic_Psi_in_plane(tau_edge, a0, R_y),
            }
        return cache[name]

    return get


MAIN_CASES = ["a10", "a30-curved"]

##################################################
# The analytic solution itself
##################################################


@pytest.mark.parametrize("name", MAIN_CASES)
def test_analytic_solution_is_consistent(name):
    """The helpers satisfy H = 0, the constraint (A.3)
    grad_K H . Psi + grad H = 0, the beam-frame edge values, and agree with
    the closed form of the beam-frame Psi_yy"""
    a0_deg, R_y, _ = CASES[name]
    a0 = np.deg2rad(a0_deg)
    tau_leave = 2 * K0 * L * np.cos(a0)
    tau = np.linspace(0, tau_leave, 101)
    q, K = analytic_ray(tau, a0)
    Psi = analytic_Psi_in_plane(tau, a0, R_y)

    # H = K^2 / K0^2 - 1 + X / L = 0, and the ray leaves at X = 0
    assert_allclose(np.sum(K**2, axis=1) / K0**2 - 1 + q[:, 0] / L, 0, atol=1e-14)
    assert abs(q[-1, 0]) < 1e-14 * L

    # (A.3): (2 K / K0^2) . Psi = -X_hat / L, at the edge and along the ray
    constraint = np.einsum("ti,tij->tj", 2 * K / K0**2, Psi)
    expected = np.broadcast_to([-1 / L, 0, 0], constraint.shape)
    assert_allclose(constraint, expected, atol=1e-12 / L)

    # (A.10), (A.11) at the edge: Psi_yy = K0 / R_y + 2i / W0^2
    assert_allclose(Psi_yy_from_lab(Psi[:1], K[:1])[0], edge_Psi_yy(R_y), rtol=1e-13)
    # The closed form of Psi_yy(K_X) and the matrix route agree
    assert_allclose(
        Psi_yy_closed_form(tau, a0, R_y), Psi_yy_from_lab(Psi, K), rtol=1e-12
    )


@pytest.mark.parametrize("name", MAIN_CASES)
def test_analytic_Psi_ZZ_is_consistent(name):
    """The closed form of Psi_ZZ for the magnetised O-mode agrees with integrating
    dPsi_ZZ/dtau = -2 (1 - Omega_pe^2 / Omega^2) Psi_ZZ^2 / K0^2"""
    a0_deg, R_y, _ = CASES[name]
    a0 = np.deg2rad(a0_deg)
    tau_leave = 2 * K0 * L * np.cos(a0)
    tau = np.linspace(0, tau_leave, 101)

    def rhs(t, y):
        Psi_ZZ = y[0] + 1j * y[1]
        density_ratio = analytic_ray(t, a0)[0][0, 0] / L  # Omega_pe^2 / Omega^2
        d = -2 * (1 - density_ratio) * Psi_ZZ**2 / K0**2
        return [d.real, d.imag]

    Psi_ZZ_0 = edge_Psi_yy(R_y)
    solution = solve_ivp(
        rhs,
        (0, tau_leave),
        [Psi_ZZ_0.real, Psi_ZZ_0.imag],
        t_eval=tau,
        rtol=1e-11,
        atol=1e-9,
    )
    assert_allclose(
        solution.y[0] + 1j * solution.y[1],
        analytic_Psi_ZZ(tau, a0, R_y),
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
    and the zero entries coupling the plane to Z"""
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
def test_Psi_ZZ_magnetised_curvature(runs, name):
    """Correcting the paper: Psi_ZZ, along B, with the magnetised O-mode's
    d^2 H / dK_Z^2 = 2 (1 - Omega_pe^2 / Omega^2) / K0^2 (module notes)"""
    run = runs(name)
    expected = analytic_Psi_ZZ(run["tau_edge"], run["a0"], run["R_y"])
    assert_allclose(
        run["Psi"][:, 2, 2],
        expected,
        rtol=0,
        atol=TOL["Psi_ZZ"] * np.abs(expected).max(),
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
    """W_y = sqrt(2 / Im Psi_yy) along the ray, and where the beam is
    narrowest.

    Scotty's output is sampled every `step` in tau, so the focus is located
    from a parabola through the three narrowest output points. The parabola's
    own error (up to 0.05 of a step at 10 deg, where the focus is sharp) is
    the same for Scotty's widths and the analytic widths at the same points,
    so those two are compared; the narrowest output point is also checked to
    be the one nearest the exact analytic focus"""
    run = runs(name)
    _, K = run["analytic_ray"]
    W = width(Psi_yy_from_lab(run["Psi"], run["K"]))
    W_expected = width(Psi_yy_from_lab(run["analytic_Psi"], K))
    assert_allclose(W, W_expected, rtol=TOL["W"])

    focus = analytic_focus(run["a0"], run["R_y"])
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
    Omega_ce / Omega = 0.3 and 0.1, on the converging beam"""
    a, b = runs("a30-curved"), runs("a30-curved-B0")
    assert_allclose(b["tau"], a["tau"], rtol=TOL["B0_tau"])
    assert_allclose(b["q"], a["q"], rtol=0, atol=TOL["B0_ray"] * L)
    assert_allclose(b["K"], a["K"], rtol=0, atol=TOL["B0_ray"] * K0)
    scale = np.abs(a["Psi"]).max()
    assert_allclose(b["Psi"], a["Psi"], rtol=0, atol=TOL["B0_Psi"] * scale)
