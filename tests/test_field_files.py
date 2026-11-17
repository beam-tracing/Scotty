"""The field constructors that read files (4.0.0): they take the file and,
for a file with several times, the time, in place of v3's ``find_B_method``
"test", "test_notime", "UDA_saved" and "EFITpp" with ``magnetic_data_path``,
``shot`` and ``equil_time``. The fields themselves are checked against the
analytical field in tests/test_simple_golden.py (``test_create_magnetic_geometry``,
``test_EFIT_geometry`` and the integrated runs)"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from scotty.geometry import EFITField, InterpolatedField_Cylindrical

R = np.linspace(1.0, 2.0, 21)
Z = np.linspace(-0.5, 0.5, 23)
R_GRID, Z_GRID = np.meshgrid(R, Z, indexing="ij")
PSI = ((R_GRID - 1.5) ** 2 + Z_GRID**2) / 0.25
TIMES = np.array([0.1, 0.2, 0.3])


def at_times(value):
    """``value`` times 1, 2 and 3: a different field at each time"""
    return np.stack([value * (1 + i) for i in range(len(TIMES))])


def B_grids_file(path, with_time):
    grids = {
        "Br_grid": 0.1 * Z_GRID,
        "Bphi_grid": 1.5 / R_GRID,
        "Bz_grid": 0.1 * R_GRID,
    }
    if with_time:
        grids = {key: at_times(value) for key, value in grids.items()}
        grids["poloidalFlux_grid"] = np.stack([PSI] * len(TIMES))
        grids["time_EFIT"] = TIMES
    else:
        grids["poloidalFlux_grid"] = PSI
    np.savez(path / "field.npz", R_EFIT=R, Z_EFIT=Z, **grids)
    return path / "field.npz"


def MAST_U_file(path, with_time, with_poloidalFlux=True):
    rBphi = np.linspace(1.5, 1.4, 11)
    data = dict(
        rBphi=rBphi,
        poloidalFlux_grid=PSI,
        poloidalFlux_unnormalised_axis=np.array(0.0),
        poloidalFlux_unnormalised_boundary=np.array(-0.1),
    )
    if with_poloidalFlux:
        # Not evenly spaced. With several times, rBphi is on it (as in v3)
        data["poloidalFlux"] = np.linspace(0, 1, 11) ** 2
    if with_time:
        # The normalised fluxes are the same at each time; rBphi and the
        # unnormalised flux, so B_T and B_p, scale with the time index
        same = ("poloidalFlux_grid", "poloidalFlux")
        data = {
            key: np.stack([value] * len(TIMES)) if key in same else at_times(value)
            for key, value in data.items()
        }
        data["time_EFIT"] = TIMES
    np.savez(path / "field.npz", R_EFIT=R, Z_EFIT=Z, **data)
    return path / "field.npz"


@pytest.mark.parametrize("time", [0.2, 0.23, 0.17])
def test_from_npz_takes_the_nearest_time(tmp_path, time):
    field = InterpolatedField_Cylindrical.from_npz(B_grids_file(tmp_path, True), time)
    # The field at 0.2 is twice the field at index 0
    assert_allclose(field.B_T(1.5, 0, 0.0), 2 * 1.5 / 1.5, rtol=1e-10)


def test_from_npz_with_one_time(tmp_path):
    field = InterpolatedField_Cylindrical.from_npz(B_grids_file(tmp_path, False))
    assert_allclose(field.B_T(1.25, 0, 0.1), 1.5 / 1.25, rtol=1e-6)
    assert_allclose(field.polflux(1.5, 0, 0.0), 0.0, atol=1e-10)


def test_from_MAST_U_saved_takes_the_nearest_time(tmp_path):
    filename = MAST_U_file(tmp_path, True)
    one = EFITField.from_MAST_U_saved(filename, 0.1)
    three = EFITField.from_MAST_U_saved(filename, 0.31)
    # psi_boundary - psi_axis scales with the time index, and so does B_p
    R0, Z0 = 1.7, 0.1
    assert_allclose(three.B_R(R0, 0, Z0), 3 * one.B_R(R0, 0, Z0), rtol=1e-10)
    assert_allclose(three.B_T(R0, 0, Z0), 3 * one.B_T(R0, 0, Z0), rtol=1e-10)


@pytest.mark.parametrize("with_poloidalFlux", [True, False])
def test_from_MAST_U_saved_with_one_time(tmp_path, with_poloidalFlux):
    """v3's "UDA_saved" with no shot: rBphi is on a flux evenly spaced from 0
    to 1, as in v3. A ``poloidalFlux`` in the file is not used: in the MAST-U
    files saved from UDA it is the flux along the midplane (1 -> 0 -> 1), not
    the flux of rBphi"""
    filename = MAST_U_file(tmp_path, False, with_poloidalFlux=False)
    if with_poloidalFlux:
        data = dict(np.load(filename))
        data["poloidalFlux"] = np.abs(np.linspace(-1, 1, 11))
        np.savez(filename, **data)
    field = EFITField.from_MAST_U_saved(filename)
    psi = float(field.polflux(1.7, 0, 0.1))
    rBphi = np.interp(psi, np.linspace(0, 1, 11), np.linspace(1.5, 1.4, 11))
    assert_allclose(field.B_T(1.7, 0, 0.1), rBphi / 1.7, rtol=1e-3)


@pytest.mark.parametrize(
    ("constructor", "make_file"),
    [
        (InterpolatedField_Cylindrical.from_npz, B_grids_file),
        (EFITField.from_MAST_U_saved, MAST_U_file),
    ],
    ids=["from_npz", "from_MAST_U_saved"],
)
def test_the_time_must_match_the_file(tmp_path, constructor, make_file):
    with pytest.raises(ValueError, match="has several times.*give `equil_time`"):
        constructor(make_file(tmp_path, True))
    with pytest.raises(ValueError, match="has one time.*leave `equil_time` out"):
        constructor(make_file(tmp_path, False), 0.2)


def test_from_MAST_saved(tmp_path):
    """MAST layout: the unnormalised flux on time base B, stored as (time, Z,
    R); the flux functions on time base C, with other times. The grid is not
    square and the flux is not symmetric in R and Z, so that a transposed
    grid would not pass"""
    psi = PSI + 0.3 * (R_GRID - 1.5)  # shifted, so not symmetric in R - 1.5, Z
    t_base_B = np.array([0.1, 0.2, 0.3])
    t_base_C = np.array([0.05, 0.1, 0.15, 0.2, 0.25])
    rBphi = np.linspace(1.5, 1.4, 11)
    axis, boundary = 0.02, -0.08
    C = np.zeros(len(t_base_C))
    C[3] = 1  # only 0.2 on time base C is the right time
    np.savez(
        tmp_path / "29999_equilibrium_data.npz",
        rBphi=np.outer(C, rBphi),
        t_base_B=t_base_B,
        t_base_C=t_base_C,
        R_EFIT=R,
        Z_EFIT=Z,
        poloidal_flux_unnormalised_axis=C * axis,
        poloidal_flux_unnormalised_boundary=C * boundary,
        poloidal_flux_unnormalised=np.stack([np.zeros_like(psi.T)] * 3)
        + np.array([0, 1, 0])[:, None, None] * (axis + psi * (boundary - axis)).T,
    )
    field = EFITField.from_MAST_saved(tmp_path / "29999_equilibrium_data.npz", 0.21)
    expected = EFITField(R, Z, rBphi, psi, axis, boundary)
    R0, Z0 = 1.62, 0.13
    for name in ("polflux", "B_R", "B_T", "B_Z"):
        assert_allclose(
            getattr(field, name)(R0, 0, Z0),
            getattr(expected, name)(R0, 0, Z0),
            rtol=1e-10,
            err_msg=name,
        )
