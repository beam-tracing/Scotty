import numpy as np
import xarray as xr

from scotty.data_io import (
    BEAM_FROM_DBS,
    from_output_tree,
    to_output_tree,
)


def _flat():
    tau = {"tau": np.arange(3.0)}
    along = ("tau", np.arange(3.0))
    return {
        "inputs": xr.Dataset({"mode_flag": 1}),
        "solver_output": xr.Dataset({"q_R_output": along}, coords=tau),
        "analysis": xr.Dataset(
            {"theta_m": along, "H_Booker": along, "e_hat": along}, coords=tau
        ),
        "dbs_analysis": xr.Dataset(
            {"theta": along, "filterfun_m": along, "k_perp1_hat": along}, coords=tau
        ),
    }


def test_groups():
    dt = to_output_tree(**_flat())
    assert set(dt["solver"].ds.data_vars) == {"q_R_output"}
    assert set(dt["beam"].ds.data_vars) == {
        "theta_m",
        "H_Booker",
        "e_hat",
        "theta",
        "k_perp1_hat",
    }
    assert set(dt["diagnostics/dbs"].ds.data_vars) == {"filterfun_m"}
    assert set(dt.children) == {"inputs", "solver", "beam", "diagnostics"}


def test_round_trip_through_a_file(tmp_path):
    flat = _flat()
    to_output_tree(**flat).to_netcdf(tmp_path / "out.h5", engine="h5netcdf")
    back = from_output_tree(xr.open_datatree(tmp_path / "out.h5", engine="h5netcdf"))
    for name, ds in flat.items():
        assert set(back[name].data_vars) == set(ds.data_vars)
    assert back["dbs_analysis"]["theta"].dims == ("tau",)


def test_beam_from_dbs_is_not_in_analysis():
    """The DBS variables moved to `beam` don't clash with `analysis`'s"""
    assert not set(BEAM_FROM_DBS) & set(_flat()["analysis"].data_vars)
