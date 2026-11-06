import numpy as np
import numpy.testing as npt
import pytest
from scipy import constants

from scotty.boundary_conditions import find_K_plasma, find_Psi_3D_plasma
from scotty.geometry import (
    CircularFluxSurfaceField,
    MagneticField_Cartesian,
    MagneticField_Cylindrical,
)
from scotty.hamiltonian import Hamiltonian
from scotty.launch import find_O_mode_flag

# Values from a test case (as in v3's test_find_Psi_3D_plasma_discontinuous)
PSI_VACUUM_3D = np.array(
    [
        [
            22.96480274 + 25.35603703j,
            -288.34856095 - 200.18496508j,
            -76.01536319 - 83.93054303j,
        ],
        [
            -288.34856095 - 200.18496508j,
            3901.7586168 + 2724.12830862j,
            -22.50617362 - 24.84965268j,
        ],
        [
            -76.01536319 - 83.93054303j,
            -22.50617362 - 24.84965268j,
            625.89678142 + 691.06894368j,
        ],
    ]
)
K_VACUUM = np.array([-720.1663330442489, -213.22253614625006, -89.39673984646942])
K_PLASMA = np.array([-572.3610161829057, -213.22253614625006, -116.25073909098793])
DH = {
    "dH_dR": -2.2863224601654686,
    "dH_dzeta": 0.0,
    "dH_dZ": 0.40208647341111176,
    "dH_dKR": -0.0021297263342362482,
    "dH_dKzeta": -0.00018858310699032543,
    "dH_dKZ": -0.00044100184796613817,
}
DPOLFLUX_DR = 1.9677859515843108
DPOLFLUX_DZ = -0.3575170608158262
D2POLFLUX_DR2 = 0.12782996883231587
D2POLFLUX_DZ2 = 3.8721770234673154
D2POLFLUX_DRDZ = 0.7035172444602721


class FluxDerivatives(MagneticField_Cylindrical):
    """Only the poloidal flux derivatives at the entry point"""

    def B_R(self, R, zeta, Z):
        raise NotImplementedError

    B_T = B_Z = polflux = B_R

    def d_polflux_dR(self, R, zeta, Z, delta_R):
        return DPOLFLUX_DR

    def d_polflux_dZ(self, R, zeta, Z, delta_Z):
        return DPOLFLUX_DZ

    def d2_polflux_dR2(self, R, zeta, Z, delta_R):
        return D2POLFLUX_DR2

    def d2_polflux_dZ2(self, R, zeta, Z, delta_Z):
        return D2POLFLUX_DZ2

    def d2_polflux_dRdZ(self, R, zeta, Z, delta_R, delta_Z):
        return D2POLFLUX_DRDZ


class HamiltonianDerivatives:
    """Only the Hamiltonian's derivatives in the plasma at the entry point"""

    spacings = {"q0": 1e-4, "q1": 1e-4, "q2": 1e-4}

    def derivatives(self, q, K):
        return dict(DH)


def test_find_Psi_3D_plasma_discontinuous():
    Psi_3D_plasma = find_Psi_3D_plasma(
        "discontinuous",
        np.array([1.0, 0.0, 0.0]),
        K_VACUUM,
        K_PLASMA,
        PSI_VACUUM_3D,
        FluxDerivatives(),
        HamiltonianDerivatives(),
    )

    # The matching conditions (Hall-Chen, Parra & Hillesheim 2022, App. C,
    # with the second-order displacement eta on the poloidal condition and the
    # sign of its cross term corrected), checked directly rather than against
    # stored values
    dpR, dpZ = DPOLFLUX_DR, DPOLFLUX_DZ
    eta = (
        -0.5
        * (
            D2POLFLUX_DR2 * dpZ**2
            - 2 * D2POLFLUX_DRDZ * dpR * dpZ
            + D2POLFLUX_DZ2 * dpR**2
        )
        / (dpR**2 + dpZ**2)
    )
    Pv, Pp = PSI_VACUUM_3D, Psi_3D_plasma  # (R, zeta, Z)
    poloidal = lambda P: P[0, 0] * dpZ**2 - 2 * P[0, 2] * dpR * dpZ + P[2, 2] * dpR**2
    mixed = lambda P: -P[0, 1] * dpZ + P[1, 2] * dpR
    dK_dot_gradpsi = (K_VACUUM[0] - K_PLASMA[0]) * dpR + (
        K_VACUUM[2] - K_PLASMA[2]
    ) * dpZ
    npt.assert_allclose(Pp[1, 1], Pv[1, 1])
    npt.assert_allclose(poloidal(Pp), poloidal(Pv) + 2 * eta * dK_dot_gradpsi)
    npt.assert_allclose(mixed(Pp), mixed(Pv))
    g_K = np.array([DH["dH_dKR"], DH["dH_dKzeta"], DH["dH_dKZ"]])
    npt.assert_allclose(
        Pp @ g_K, -np.array([DH["dH_dR"], 0.0, DH["dH_dZ"]]), atol=1e-12
    )
    npt.assert_allclose(Pp, Pp.T)


##################################################
# Cartesian entry points where grad psi is along a coordinate axis
##################################################


# Hessian of psi at the entry point, and the Hamiltonian's derivatives there
_HESS_PSI = np.array([[0.7, 0.2, -0.3], [0.2, 1.1, 0.4], [-0.3, 0.4, 2.3]])
_DH_CART = {
    "dH_dX": -2.3,
    "dH_dY": 0.6,
    "dH_dZ": 0.4,
    "dH_dKX": -2.1e-3,
    "dH_dKY": 3.0e-4,
    "dH_dKZ": -4.4e-4,
}


class FluxDerivativesCartesian(MagneticField_Cartesian):
    """Only the poloidal flux derivatives at the entry point"""

    def __init__(self, grad_psi):
        self.grad_psi = grad_psi

    def B_X(self, X, Y, Z):
        raise NotImplementedError

    B_Y = B_Z = polflux = B_X

    def d_polflux_dX(self, X, Y, Z, delta_X):
        return self.grad_psi[0]

    def d_polflux_dY(self, X, Y, Z, delta_Y):
        return self.grad_psi[1]

    def d_polflux_dZ(self, X, Y, Z, delta_Z):
        return self.grad_psi[2]

    def d2_polflux_dX2(self, X, Y, Z, delta_X):
        return _HESS_PSI[0, 0]

    def d2_polflux_dY2(self, X, Y, Z, delta_Y):
        return _HESS_PSI[1, 1]

    def d2_polflux_dZ2(self, X, Y, Z, delta_Z):
        return _HESS_PSI[2, 2]

    def d2_polflux_dXdY(self, X, Y, Z, delta_X, delta_Y):
        return _HESS_PSI[0, 1]

    def d2_polflux_dXdZ(self, X, Y, Z, delta_X, delta_Z):
        return _HESS_PSI[0, 2]

    def d2_polflux_dYdZ(self, X, Y, Z, delta_Y, delta_Z):
        return _HESS_PSI[1, 2]


class HamiltonianDerivativesCartesian:
    spacings = {"q0": 1e-4, "q1": 1e-4, "q2": 1e-4}

    def derivatives(self, q, K):
        return dict(_DH_CART)


@pytest.mark.parametrize("Psi_BC_flag", ["continuous", "discontinuous"])
@pytest.mark.parametrize(
    "grad_psi",
    [
        [1.9, 0.0, 0.0],
        [0.0, 1.9, 0.0],
        [0.0, 0.0, 1.9],
        [1.9, -1.9, 0.5],
        [1.9, 0.0, -0.4],
        [1.9, 0.5, -0.4],
    ],
    ids=["along-X", "along-Y", "along-Z", "X-equals-minus-Y", "XZ-plane", "general"],
)
def test_find_Psi_3D_plasma_any_normal(Psi_BC_flag, grad_psi):
    """The matching conditions hold whatever the direction of grad psi. The
    first four (along a coordinate axis, as in a slab or at a midplane entry,
    or with psi_X = -psi_Y) made the old coordinate-plane interface matrix
    singular."""
    grad_psi = np.array(grad_psi)
    K_vacuum = np.array([-720.2, -213.2, -89.4])
    if Psi_BC_flag == "continuous":
        K_plasma = K_vacuum
    else:  # K jumps along grad psi only
        K_plasma = K_vacuum + 75.0 * grad_psi
    Psi_v = PSI_VACUUM_3D
    Psi_p = find_Psi_3D_plasma(
        Psi_BC_flag,
        np.array([1.0, 0.0, 0.0]),
        K_vacuum,
        K_plasma,
        Psi_v,
        FluxDerivativesCartesian(grad_psi),
        HamiltonianDerivativesCartesian(),
    )

    # t.Psi.t jumps by -(t.(grad grad psi).t) (delta_K . grad psi) / |grad psi|^2
    # along every tangent t, here 5 random ones
    rng = np.random.default_rng(0)
    for t in rng.normal(size=(5, 3)):
        t = t - (t @ grad_psi) / (grad_psi @ grad_psi) * grad_psi
        jump = -(t @ _HESS_PSI @ t) * ((K_vacuum - K_plasma) @ grad_psi)
        jump /= grad_psi @ grad_psi
        npt.assert_allclose(t @ Psi_p @ t, t @ Psi_v @ t + jump, rtol=1e-10)
    g_K = np.array([_DH_CART["dH_dKX"], _DH_CART["dH_dKY"], _DH_CART["dH_dKZ"]])
    g_q = np.array([_DH_CART["dH_dX"], _DH_CART["dH_dY"], _DH_CART["dH_dZ"]])
    npt.assert_allclose(Psi_p @ g_K, -g_q, atol=1e-12)
    npt.assert_allclose(Psi_p, Psi_p.T)


##################################################
# Which cut-off is checked at the plasma boundary (discontinuous BC)
##################################################


_OMEGA = 2 * np.pi * 55e9
_R_ENTRY = 2.0
# Beam below the cyclotron frequency at the entry (Y = w_ce / w = 1.5), where
# the O-mode is `mode_flag` = -1 and +1 is the X-mode
_Y = 1.5
_FIELD = CircularFluxSurfaceField(
    _Y * _OMEGA * constants.m_e / constants.e * _R_ENTRY / 1.5, 1.5, 0.5, 0.01
)


def _jump_to(X):
    """Density that jumps at the boundary (psi = 1) to w_pe^2 / w^2 = X"""
    n_e = X * _OMEGA**2 * constants.epsilon_0 * constants.m_e / constants.e**2 / 1e19
    return lambda psi: np.where(np.asarray(psi) < 1.0, n_e, 0.0)


def _find_K_plasma(X, mode_flag):
    q_entry = np.array([_R_ENTRY - 1e-9, 0.0, 0.0])
    K_vacuum = np.array([-_OMEGA / constants.c, 0.0, 0.0])
    hamiltonian = Hamiltonian(
        _OMEGA, mode_flag, np.array([-1e-4, 1e-4, 0.1, 0.1, 0.1]), _FIELD, _jump_to(X)
    )
    assert (
        find_O_mode_flag(q_entry, K_vacuum, _FIELD, _OMEGA, hamiltonian.density) == -1
    )
    return find_K_plasma("discontinuous", q_entry, K_vacuum, _FIELD, hamiltonian)


def test_X_mode_at_entry_is_not_checked_against_the_O_mode_cut_off():
    """At X = 1.2 the O-mode is cut off (w_pe > w) but the X-mode is not
    (w_L < w < w_UH). `mode_flag` = +1 is the X-mode here, so it enters; it used
    to be taken as the O-mode, and refused"""
    K_plasma = _find_K_plasma(1.2, mode_flag=1)
    assert np.all(np.isfinite(K_plasma))


def test_O_mode_at_entry_checked_against_the_plasma_frequency():
    with pytest.raises(
        ValueError, match=r"for the O-mode \(`mode_flag` = -1\): w_plasma"
    ):
        _find_K_plasma(1.2, mode_flag=-1)


def test_X_mode_at_entry_checked_against_the_L_cut_off():
    """At X = 3, w_L > w: the X-mode (`mode_flag` = +1 here) is cut off"""
    with pytest.raises(ValueError, match=r"for the X-mode \(`mode_flag` = 1\): w_L"):
        _find_K_plasma(3.0, mode_flag=1)
