"""The discontinuous boundary condition keeps the beam's phase continuous across
the plasma boundary to second order in the displacement along the boundary.

Two entry points, on elliptical flux surfaces, with a density jump at the
boundary so that K jumps along grad psi and eta matters:
- "tilted": tilted ellipses, entered off the ellipse's axes, so psi_RZ != 0.
  This tests the cross term of eta (Hall-Chen, Parra & Hillesheim 2022, eq.
  C.6, prints the wrong sign for it); the analytic circular fields have
  psi_RZ = 0 everywhere.
- "midplane": untilted ellipses, entered at Z = 0, so psi_Z = psi_RZ = 0 and
  the sign of the cross term does not matter, but eta = -psi_ZZ / 2 (the
  surface's curvature) does not vanish. This tests that eta goes into the
  poloidal matching condition, not the mixed one."""

import numpy as np
import pytest
from scipy.optimize import brentq

from scotty.beam_me_up import beam_me_up
from scotty.fun_general import find_K_lab, find_K_lab_Cartesian
from scotty.geometry import InterpolatedField
from scotty.init_bruv import get_parameters_for_Scotty
from scotty.profile_fit import QuadraticFit

R0, A, B = 1.5, 0.5, 0.8

CASES = {
    # tilt (rad), launch position (R, zeta, Z), poloidal launch angle (deg)
    "tilted": (0.6, [2.45, 0.0, 0.9], 25.0),
    "midplane": (0.0, [2.45, 0.0, 0.0], 0.0),
}


def tilted_ellipse_field(tilt):
    R = np.linspace(0.6, 2.6, 201)
    Z = np.linspace(-1.2, 1.2, 241)
    RR, ZZ = np.meshgrid(R, Z, indexing="ij")
    u = (RR - R0) * np.cos(tilt) + ZZ * np.sin(tilt)
    v = -(RR - R0) * np.sin(tilt) + ZZ * np.cos(tilt)
    psi = u**2 / A**2 + v**2 / B**2
    psi_R = 2 * u * np.cos(tilt) / A**2 - 2 * v * np.sin(tilt) / B**2
    psi_Z = 2 * u * np.sin(tilt) / A**2 + 2 * v * np.cos(tilt) / B**2
    c = 0.05
    return InterpolatedField(R, Z, -c * psi_Z / RR, R0 / RR, c * psi_R / RR, psi, 5, 0)


@pytest.fixture(scope="module", params=list(CASES))
def boundary(request, tmp_path_factory):
    tilt, launch, poloidal_angle = CASES[request.param]
    field = tilted_ellipse_field(tilt)
    kwargs = get_parameters_for_Scotty("DBS_synthetic")
    for name in ("B_T_axis", "B_p_a", "R_axis", "minor_radius_a"):
        kwargs.pop(name, None)
    launch_position = np.array(launch)
    kwargs.update(
        find_B_method=field,
        len_tau=52,
        interp_smoothing=0,
        Psi_BC_flag="discontinuous",
        # zero at psi = 1.3, but cut to zero at the boundary psi = 1: a jump
        density_fit_method=QuadraticFit(1.3, 4.0),
        poloidal_flux_enter=1.0,
        poloidal_flux_zero_density=1.0,
        launch_position=launch_position,
        poloidal_launch_angle_Torbeam=poloidal_angle,
        delta_R=-1e-5,
        delta_Z=-1e-5,
        figure_flag=False,
        output_path=str(tmp_path_factory.mktemp("bc")) + "/",
    )
    dt = beam_me_up(**kwargs)
    inputs = dt["inputs"]
    q = inputs["initial_position"].values
    # in vacuum K is constant in Cartesian; launch_K is cylindrical at the launch position
    K_launch_cart = find_K_lab_Cartesian(inputs["launch_K"].values, launch_position)
    q_cart = np.array([q[0] * np.cos(q[1]), q[0] * np.sin(q[1]), q[2]])
    K_v = find_K_lab(K_launch_cart, q_cart)
    return dict(
        case=request.param,
        field=field,
        q=q,
        dK=inputs["K_initial"].values - K_v,
        dPsi=inputs["Psi_3D_lab_initial"].values
        - dt["analysis"]["Psi_3D_lab_entry"].values,
    )


@pytest.mark.parametrize(
    "toroidal_per_poloidal", [0.0, 0.3], ids=["poloidal", "poloidal-and-toroidal"]
)
def test_phase_continuous_to_second_order(boundary, toroidal_per_poloidal):
    R, zeta, Z = boundary["q"]
    f = lambda r, z: float(boundary["field"].poloidal_flux(r, z))
    h = 1e-5
    psi_R = (f(R + h, Z) - f(R - h, Z)) / (2 * h)
    psi_Z = (f(R, Z + h) - f(R, Z - h)) / (2 * h)
    normal = np.array([psi_R, psi_Z]) / np.hypot(psi_R, psi_Z)
    psi_RZ = (f(R + h, Z + h) - f(R + h, Z - h) - f(R - h, Z + h) + f(R - h, Z - h)) / (
        4 * h**2
    )
    psi_RR = (f(R + h, Z) - 2 * f(R, Z) + f(R - h, Z)) / h**2
    psi_ZZ = (f(R, Z + h) - 2 * f(R, Z) + f(R, Z - h)) / h**2
    eta = (
        -0.5
        * (psi_RR * psi_Z**2 - 2 * psi_RZ * psi_R * psi_Z + psi_ZZ * psi_R**2)
        / (psi_R**2 + psi_Z**2)
    )
    if boundary["case"] == "tilted":
        assert abs(psi_RZ * psi_R * psi_Z) > 1, "the entry must test the cross term"
    else:
        assert abs(psi_Z) < 1e-6 and abs(psi_RZ) < 1e-3, "the entry must be at Z = 0"
        assert abs(eta) > 1, "the surface must be curved there, so that eta != 0"
    assert np.linalg.norm(boundary["dK"]) > 10, "K must jump for eta to matter"

    xis = np.logspace(-4, -2, 6) / np.hypot(psi_R, psi_Z)
    mismatch = []
    for xi in xis:
        # a point exactly on the boundary surface, a poloidal distance ~xi |grad psi| away
        first_order = np.array([R - psi_Z * xi, Z + psi_R * xi])
        s = brentq(
            lambda s: f(*(first_order + s * normal)) - f(R, Z), -0.1 * xi, 0.1 * xi
        )
        dR, dZ = first_order + s * normal - np.array([R, Z])
        dq = np.array([dR, toroidal_per_poloidal * xi, dZ])
        mismatch.append(abs(boundary["dK"] @ dq + 0.5 * dq @ boundary["dPsi"] @ dq))
    slope = np.polyfit(np.log(xis), np.log(mismatch), 1)[0]
    assert (
        slope > 2.8
    ), f"phase mismatch ~ xi^{slope:.2f}; continuity to second order needs xi^3"
