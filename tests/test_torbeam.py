from scotty.torbeam import Torbeam

import numpy as np
import numpy.testing as npt


def test_torbeam_roundtrip(tmp_path):
    R_grid = np.linspace(0.1, 1)
    Z_grid = np.linspace(-1, 1)
    R_meshgrid, Z_meshgrid = np.meshgrid(R_grid, Z_grid, indexing="ij")
    B_r = R_meshgrid
    B_t = 1 / R_meshgrid
    B_z = Z_meshgrid
    psi = np.sqrt(R_meshgrid**2 + Z_meshgrid**2)

    torbeam = Torbeam(R_grid, Z_grid, B_r, B_t, B_z, psi)

    filename = tmp_path / "topfile"
    torbeam.write(filename)

    torbeam2 = Torbeam.from_file(filename)

    npt.assert_allclose(torbeam.R_grid, torbeam2.R_grid)
    npt.assert_allclose(torbeam.Z_grid, torbeam2.Z_grid)
    npt.assert_allclose(torbeam.B_R, torbeam2.B_R)
    npt.assert_allclose(torbeam.B_T, torbeam2.B_T)
    npt.assert_allclose(torbeam.B_Z, torbeam2.B_Z)
    npt.assert_allclose(torbeam.psi, torbeam2.psi)
