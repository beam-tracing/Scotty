from scotty.torbeam import Torbeam

import numpy as np
import numpy.testing as npt
import pytest


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


def test_torbeam_newer_headers(tmp_path):
    # Newer topfiles label the sections differently
    R_grid = np.linspace(0.1, 1)
    Z_grid = np.linspace(-1, 1)
    R_meshgrid, Z_meshgrid = np.meshgrid(R_grid, Z_grid, indexing="ij")
    torbeam = Torbeam(
        R_grid, Z_grid, R_meshgrid, 1 / R_meshgrid, Z_meshgrid, R_meshgrid * Z_meshgrid
    )

    filename = tmp_path / "topfile"
    torbeam.write(filename)
    text = filename.read_text()
    for old, new in (
        ("X-coordinates", "Radial grid coordinates"),
        ("Z-coordinates", "Vertical grid coordinates"),
        ("B_R", "B_r"),
        ("B_Z", "B_z"),
    ):
        text = text.replace(old, new)
    filename.write_text(text)

    torbeam2 = Torbeam.from_file(filename)

    npt.assert_allclose(torbeam.R_grid, torbeam2.R_grid)
    npt.assert_allclose(torbeam.Z_grid, torbeam2.Z_grid)
    npt.assert_allclose(torbeam.B_R, torbeam2.B_R)
    npt.assert_allclose(torbeam.B_T, torbeam2.B_T)
    npt.assert_allclose(torbeam.B_Z, torbeam2.B_Z)
    npt.assert_allclose(torbeam.psi, torbeam2.psi)


def test_torbeam_no_grid_header(tmp_path):
    # Used to loop forever at the end of the file
    filename = tmp_path / "topfile"
    filename.write_text("Dummy line\n1 2\n")
    with pytest.raises(ValueError, match="No radial grid header"):
        Torbeam.from_file(filename)


def test_torbeam_main_writes_inputs(tmp_path):
    # main() writes ne.dat, Te.dat, inbeam.dat and a topfile of the
    # analytical circular field, which reads back as that field
    from scotty.geometry import CircularFluxSurfaceField
    from scotty.torbeam import main

    main(B_poloidal_max=0.1, torbeam_directory_path=tmp_path)
    for name in ("ne.dat", "Te.dat", "inbeam.dat", "topfile"):
        assert (tmp_path / name).is_file(), name

    torbeam = Torbeam.from_file(tmp_path / "topfile")
    field = CircularFluxSurfaceField(1.0, 0.75, 0.5, 0.1, 130, 65, 1.1)
    R, Z = np.meshgrid(field.R_coord, field.Z_coord, indexing="ij")
    npt.assert_allclose(torbeam.R_grid, field.R_coord)
    npt.assert_allclose(torbeam.Z_grid, field.Z_coord)
    npt.assert_allclose(torbeam.B_R, field.B_R(R, 0.0, Z), atol=1e-12)
    npt.assert_allclose(torbeam.B_T, field.B_T(R, 0.0, Z))
    npt.assert_allclose(torbeam.B_Z, field.B_Z(R, 0.0, Z), atol=1e-12)
    npt.assert_allclose(torbeam.psi, field.polflux(R, 0.0, Z))
