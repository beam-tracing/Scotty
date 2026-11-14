"""A Cartesian slab with a vacuum launch, through `beam_me_up`, with both boundary
conditions (beam-tracing/Scotty#176).

The field is uniform, B = B0 Y_hat, and psi is linear along one coordinate axis, so
grad psi at the plasma entry point has only one nonzero component. Before #176, both
boundary conditions failed there: `find_Psi_3D_plasma`'s interface matrix was singular
(grad psi along X or Z), and with the discontinuous one `find_K_plasma`'s normal was
NaN (grad psi along Z).

The flux surfaces are flat, so the jump conditions are simple: K jumps only along
grad psi, and the block of Psi along the surface is the same on both sides. The slab
with grad psi along Z is the one with grad psi along X turned by 90 deg about Y, with
the launch turned with it, so the two runs must agree.
"""

import numpy as np
import pytest
from scipy import constants

from scotty.beam_me_up import beam_me_up
from scotty.geometry import MagneticField_Cartesian

FREQ = 55e9
L = 0.14
B0 = 1.0
N_CUTOFF_19 = (
    constants.epsilon_0
    * constants.m_e
    * (2 * np.pi * FREQ) ** 2
    / constants.e**2
    / 1e19
)


class Slab(MagneticField_Cartesian):
    """B = B0 Y_hat and psi = q[axis] / L: the plasma edge (psi = 1) is the plane
    q[axis] = L, with the plasma on the side of smaller q[axis]"""

    def __init__(self, axis):
        self.axis = axis
        normal_range, other_range = np.linspace(0.02, 0.5, 3), np.linspace(-1, 1, 3)
        self.X_coord = normal_range if axis == 0 else other_range
        self.Y_coord = other_range
        self.Z_coord = normal_range if axis == 2 else other_range

    @staticmethod
    def _zero(X):
        return np.zeros_like(np.asarray(X, dtype=float))

    def B_X(self, X, Y, Z):
        return self._zero(X)

    def B_Y(self, X, Y, Z):
        return B0 + self._zero(X)

    def B_Z(self, X, Y, Z):
        return self._zero(X)

    def polflux(self, X, Y, Z):
        return np.asarray((X, Y, Z)[self.axis], dtype=float) / L + self._zero(X)

    def _gradient(self, axis, X):
        return (1 / L if axis == self.axis else 0.0) + self._zero(X)

    def d_polflux_dX(self, X, Y, Z, delta_X=None):
        return self._gradient(0, X)

    def d_polflux_dY(self, X, Y, Z, delta_Y=None):
        return self._gradient(1, X)

    def d_polflux_dZ(self, X, Y, Z, delta_Z=None):
        return self._gradient(2, X)

    def d2_polflux_dX2(self, X, Y, Z, delta_X=None):
        return self._zero(X)

    def d2_polflux_dY2(self, X, Y, Z, delta_Y=None):
        return self._zero(X)

    def d2_polflux_dZ2(self, X, Y, Z, delta_Z=None):
        return self._zero(X)

    def d2_polflux_dXdY(self, X, Y, Z, delta_X=None, delta_Y=None):
        return self._zero(X)

    def d2_polflux_dXdZ(self, X, Y, Z, delta_X=None, delta_Z=None):
        return self._zero(X)

    def d2_polflux_dYdZ(self, X, Y, Z, delta_Y=None, delta_Z=None):
        return self._zero(X)


# Where the density is set to zero. With the discontinuous boundary condition it is
# just outside the edge (psi = 1), where the beam enters, so that the density there
# is the jump's, not zero
ZERO_DENSITY = {"continuous": 1.0, "discontinuous": 1.01}


def density(Psi_BC_flag):
    """Linear in psi: 0 at the edge (continuous), or 0.2 n_c there, a jump from the
    vacuum (discontinuous), rising by 0.6 n_c per unit of psi inwards"""
    jump = 0.2 if Psi_BC_flag == "discontinuous" else 0.0

    def n_e(psi):
        psi = np.asarray(psi, dtype=float)
        return N_CUTOFF_19 * np.where(
            psi < ZERO_DENSITY[Psi_BC_flag], jump + 0.6 * (1 - psi), 0.0
        )

    return n_e


def run(axis, Psi_BC_flag, output_path):
    """grad psi along X: launched from X = 0.3 towards -X, 10 deg below the
    horizontal. Along Z: the same turned by 90 deg about Y, from Z = 0.3 towards
    -Z, 10 deg towards +X. Both at Y = 0.2, off the Z axis, where the outputs'
    cylindrical coordinates would divide by R = 0.

    The toroidal angles are measured from the local R at the antenna, at
    zeta_a = atan2(Y, X), so zeta_a is taken off the angles from the X axis"""
    launch = {
        0: dict(
            launch_position=np.array([0.3, 0.2, 0.0]),
            poloidal_launch_angle_Torbeam=10.0,
            toroidal_launch_angle_Torbeam=0.0 - np.rad2deg(np.arctan2(0.2, 0.3)),
        ),
        2: dict(
            launch_position=np.array([0.0, 0.2, 0.3]),
            poloidal_launch_angle_Torbeam=80.0,
            toroidal_launch_angle_Torbeam=180.0 - np.rad2deg(np.arctan2(0.2, 0.0)),
        ),
    }[axis]
    deltas = {"delta_X": 1e-4, "delta_Y": 1e-4, "delta_Z": 1e-4}
    deltas["delta_" + "XYZ"[axis]] = -1e-4  # into the plasma
    return beam_me_up(
        geometry="cartesian",
        launch_freq_GHz=FREQ / 1e9,
        mode_flag=1,
        launch_beam_width=0.03,
        launch_beam_curvature=-1 / 0.5,
        launch_flag="vacuum_analytic",
        Psi_BC_flag=Psi_BC_flag,
        find_B_method=Slab(axis),
        density_fit_method=density(Psi_BC_flag),
        poloidal_flux_enter=1.0,
        poloidal_flux_zero_density=ZERO_DENSITY[Psi_BC_flag],
        len_tau=202,
        output_path=output_path,
        **launch,
        **deltas,
    )


class _Runs(dict):
    """Each run once, when a test first needs it, so that a run that fails fails
    only the tests that use it"""

    def __init__(self, tmp_path_factory):
        super().__init__()
        self.tmp_path_factory = tmp_path_factory

    def __missing__(self, key):
        axis, Psi_BC_flag = key
        path = self.tmp_path_factory.mktemp(f"slab_{axis}_{Psi_BC_flag}")
        self[key] = run(axis, Psi_BC_flag, path)
        return self[key]


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    return _Runs(tmp_path_factory)


@pytest.mark.parametrize("Psi_BC_flag", ["continuous", "discontinuous"])
@pytest.mark.parametrize("axis", [0, 2], ids=["grad-psi-along-X", "grad-psi-along-Z"])
def test_jump_conditions_at_a_flat_boundary(runs, axis, Psi_BC_flag):
    solver = runs[(axis, Psi_BC_flag)]["solver"]
    along = [i for i in range(3) if i != axis]
    K_vacuum = solver["K_launch_cartesian"].values
    K_plasma = solver["K_initial_cartesian"].values
    Psi_vacuum = solver["Psi_3D_entry_labframe_cartesian"].values
    Psi_plasma = solver["Psi_3D_initial_labframe_cartesian"].values

    # K along the surface is continuous; with the discontinuous boundary its normal
    # component changes, and it still points into the plasma (towards smaller psi)
    np.testing.assert_allclose(K_plasma[along], K_vacuum[along], rtol=1e-12, atol=1e-9)
    if Psi_BC_flag == "continuous":
        np.testing.assert_allclose(K_plasma, K_vacuum, rtol=1e-12)
    else:
        assert abs(K_plasma[axis]) < 0.9 * abs(K_vacuum[axis])
    assert K_plasma[axis] < 0

    # A flat surface has no curvature terms, so the block of Psi along it is the
    # same on both sides, whichever boundary condition
    block = np.ix_(along, along)
    np.testing.assert_allclose(
        Psi_plasma[block], Psi_vacuum[block], rtol=1e-10, atol=1e-9
    )


@pytest.mark.parametrize("Psi_BC_flag", ["continuous", "discontinuous"])
def test_slab_along_Z_is_the_slab_along_X_turned(runs, Psi_BC_flag):
    """The rotation by 90 deg about Y that takes X_hat to Z_hat (and Z_hat to
    -X_hat) takes the run along X to the run along Z"""
    turn = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    along_X = runs[(0, Psi_BC_flag)]["solver"]
    along_Z = runs[(2, Psi_BC_flag)]["solver"]

    np.testing.assert_allclose(
        turn @ along_X["K_initial_cartesian"].values,
        along_Z["K_initial_cartesian"].values,
        rtol=1e-9,
        atol=1e-9 * np.linalg.norm(along_X["K_initial_cartesian"].values),
    )
    Psi_X = along_X["Psi_3D_initial_labframe_cartesian"].values
    Psi_Z = along_Z["Psi_3D_initial_labframe_cartesian"].values
    np.testing.assert_allclose(
        turn @ Psi_X @ turn.T, Psi_Z, rtol=1e-8, atol=1e-8 * np.abs(Psi_X).max()
    )
    np.testing.assert_allclose(
        turn @ along_X["q_output_cartesian"].values.T,
        along_Z["q_output_cartesian"].values.T,
        rtol=0,
        atol=1e-6,
    )
