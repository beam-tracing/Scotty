"""Run Scotty v3 inputs on v4.

``from_v3_kwargs`` takes the keyword arguments a v3 script passed to
``beam_me_up`` and returns the equivalent v4 ones, logging a warning for
each change, so a v3 script needs one line::

    from scotty.compat import from_v3_kwargs
    beam_me_up(**from_v3_kwargs(kwargs_dict))

Arguments given explicitly are never overwritten, and applying it twice
gives the same result as once. Arguments left unset get v3's defaults, so
use it for v3 inputs, not v4 ones.

What it converts:

- the launch flags: ``vacuumLaunch_flag`` to ``launch_flag``, and
  ``Psi_BC_flag`` to ``boundary_flag``; ``geometry`` is set to
  ``"cylindrical"``, the only geometry v3 had;
- a plasma launch: ``plasmaLaunch_K`` (cylindrical, as in v3) to the
  Cartesian ``plasmaLaunch_K_cartesian``, and
  ``plasmaLaunch_Psi_3D_lab_Cartesian`` to
  ``plasmaLaunch_Psi_3D_lab_cartesian``;
- ``quick_run`` to ``ray_tracing_flag``; ``quick_cutoff(dt)`` then gives the
  numbers v3's ``quick_run`` returned (R, Z, |K|, poloidal flux and θ_m at the
  minimum of |K|);
- ``output_path`` given as a string, to a ``pathlib.Path``;
- defaults that changed between v3 and v4, restored when the v3 script left
  them unset: ``interp_smoothing`` (0), ``find_B_method`` ("torbeam"),
  ``detailed_analysis_flag`` (True), and ``poloidal_flux_zero_density`` and
  ``poloidal_flux_zero_temperature`` (1.01, raised to
  ``poloidal_flux_enter`` if that is larger, since v4 requires it).

``vacuum_propagation_flag`` has no v4 equivalent and is dropped with a warning.
``resonance_flag`` is passed through unchanged (v4 has it since #151 was ported).

Output files
------------

v4 also reorganised the output: renamed variables, a separate
``dbs_analysis`` group, lab-frame quantities in both cylindrical and
Cartesian coordinates, and new dimension names. ``to_v4_output`` and
``to_v3_output`` convert an output ``DataTree`` either way, and
``convert_output_file`` converts a file, detecting its version::

    from scotty.compat import convert_output_file
    convert_output_file("scotty_output_v3.h5", "scotty_output_as_v4.h5")

Only quantities with a counterpart in the other version are converted. The
others are dropped, and their names are recorded in the root attribute
``scotty_compat_not_converted``, so nothing disappears silently.
"""

import logging
import pathlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import xarray as xr

from scotty.fun_general import find_K_labframe_cyl_to_cart

log = logging.getLogger(__name__)

#: v3 arguments that v4's beam_me_up no longer takes
V3_ONLY_ARGUMENTS = (
    "vacuumLaunch_flag",
    "vacuum_propagation_flag",
    "Psi_BC_flag",
    "quick_run",
    "plasmaLaunch_K",
    "plasmaLaunch_Psi_3D_lab_Cartesian",
)

#: Defaults that changed between v3 and v4: name -> v3 default
V3_DEFAULTS = {
    "interp_smoothing": 0,
    "find_B_method": "torbeam",
    "detailed_analysis_flag": True,
}
V3_DEFAULT_POLOIDAL_FLUX_ZERO = 1.01


_warned = set()


def _warn(message: str) -> None:
    # Once per message, so that a sweep calling this for every run stays readable
    if message not in _warned:
        _warned.add(message)
        log.warning(f"from_v3_kwargs: {message}")


def from_v3_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of ``kwargs`` with v3 arguments converted to v4"""
    kw = dict(kwargs)

    # Launch from vacuum or from inside the plasma
    if "vacuumLaunch_flag" in kw:
        vacuum_launch = bool(kw.pop("vacuumLaunch_flag"))
        if "launch_flag" not in kw:
            kw["launch_flag"] = "vacuum" if vacuum_launch else "plasma"
            _warn(f"`vacuumLaunch_flag` = {vacuum_launch} -> `launch_flag` = '{kw['launch_flag']}'")
    elif "launch_flag" not in kw:
        kw["launch_flag"] = "vacuum"  # v3's default
        _warn("set `launch_flag` = 'vacuum' (v3's default `vacuumLaunch_flag` = True)")

    # Plasma-vacuum boundary conditions; booleans were already deprecated in v3
    if "Psi_BC_flag" in kw:
        Psi_BC_flag = kw.pop("Psi_BC_flag")
        if Psi_BC_flag is True:
            Psi_BC_flag = "continuous"
        elif Psi_BC_flag is False:
            Psi_BC_flag = None
        if "boundary_flag" not in kw:
            kw["boundary_flag"] = Psi_BC_flag
            _warn(f"`Psi_BC_flag` -> `boundary_flag` = {Psi_BC_flag!r}")
    elif "boundary_flag" not in kw:
        kw["boundary_flag"] = None  # v3's default
        _warn("set `boundary_flag` = None (v3's default `Psi_BC_flag` = None)")

    if "geometry" not in kw:
        kw["geometry"] = "cylindrical"
        _warn("set `geometry` = 'cylindrical' (v3 had no other geometry)")

    # Plasma launch: v3 took K in cylindrical (K_R, K_zeta, K_Z), v4 in Cartesian
    if "plasmaLaunch_K" in kw:
        K_cyl = np.asarray(kw.pop("plasmaLaunch_K"), dtype=float)
        if "plasmaLaunch_K_cartesian" not in kw and np.any(K_cyl != 0):
            q_cyl = np.asarray(kw["launch_position"], dtype=float)
            kw["plasmaLaunch_K_cartesian"] = find_K_labframe_cyl_to_cart(K_cyl, q_cyl)
            _warn("`plasmaLaunch_K` (cylindrical) -> `plasmaLaunch_K_cartesian`")
    if "plasmaLaunch_Psi_3D_lab_Cartesian" in kw:
        Psi = kw.pop("plasmaLaunch_Psi_3D_lab_Cartesian")
        if "plasmaLaunch_Psi_3D_lab_cartesian" not in kw and np.any(np.asarray(Psi) != 0):
            kw["plasmaLaunch_Psi_3D_lab_cartesian"] = Psi
            _warn("`plasmaLaunch_Psi_3D_lab_Cartesian` -> `plasmaLaunch_Psi_3D_lab_cartesian`")

    if "quick_run" in kw:
        quick_run = bool(kw.pop("quick_run"))
        if "ray_tracing_flag" not in kw:
            kw["ray_tracing_flag"] = quick_run
            _warn(f"`quick_run` -> `ray_tracing_flag` = {quick_run}")

    if "vacuum_propagation_flag" in kw:
        _warn(
            f"dropped `vacuum_propagation_flag` = {kw.pop('vacuum_propagation_flag')!r}: "
            "it has no v4 equivalent"
        )

    # v3 built output file names by string concatenation; v4 joins Paths
    if isinstance(kw.get("output_path"), str):
        output_path = kw["output_path"]
        if output_path and not output_path.endswith(("/", "\\")):
            _warn(
                f"`output_path` = '{output_path}' has no trailing separator; v3 "
                "used it as a file-name prefix, v4 uses it as a directory"
            )
        kw["output_path"] = pathlib.Path(output_path)

    for name, v3_default in V3_DEFAULTS.items():
        if name not in kw:
            kw[name] = v3_default
            _warn(f"set `{name}` = {v3_default!r} (v3's default; v4's differs)")

    enter = kw.get("poloidal_flux_enter", 1.0)
    for name in ("poloidal_flux_zero_density", "poloidal_flux_zero_temperature"):
        if name not in kw:
            kw[name] = max(V3_DEFAULT_POLOIDAL_FLUX_ZERO, enter)
            _warn(f"set `{name}` = {kw[name]} (v3's default, at least `poloidal_flux_enter`)")

    return kw


class CutoffData(NamedTuple):
    """v3's ``quick_run`` result (``K_cutoff_data``), as an iterable tuple"""

    q_R: float
    q_Z: float
    K_norm_min: float
    poloidal_flux: float
    theta_m: float


def quick_cutoff(dt: xr.DataTree) -> CutoffData:
    """The quantities v3's ``quick_run`` returned, at the minimum of |K|, from
    a v4 run (typically one with ``ray_tracing_flag=True``, which is what
    ``from_v3_kwargs`` turns ``quick_run`` into)::

        q_R, q_Z, K_norm, polflux, theta_m = quick_cutoff(beam_me_up(**from_v3_kwargs(kwargs)))
    """
    so, an = dt["solver_output"].ds, dt["analysis"].ds
    K = np.asarray(so["K_output_magnitude"])
    i = int(np.argmin(K))
    return CutoffData(
        float(so["q_R_output"][i]),
        float(so["q_Z_output"][i]),
        float(K[i]),
        float(an["polflux"][i]),
        float(an["theta_m"][i]),
    )


##################################################
# Output files
##################################################

Transform = Callable[[np.ndarray, Dict[str, np.ndarray]], np.ndarray]


def _same(a: np.ndarray, ctx: Dict[str, np.ndarray]) -> np.ndarray:
    return a


@dataclass(frozen=True)
class Rule:
    v3: Tuple[str, str]  # (group, name)
    v4: Tuple[str, str]
    v3_dims: Optional[Tuple[str, ...]] = None  # None: same dims as the source
    v4_dims: Optional[Tuple[str, ...]] = None
    to_v4: Transform = _same
    to_v3: Transform = _same


# Lab-frame vectors along the ray, (tau, 3): cylindrical (R, zeta, Z) physical
# components <-> Cartesian (X, Y, Z), at the ray's toroidal angle zeta
def _cyl_to_cart(v: np.ndarray, ctx: Dict[str, np.ndarray]) -> np.ndarray:
    c, s = np.cos(ctx["zeta"]), np.sin(ctx["zeta"])
    return np.stack([v[:, 0] * c - v[:, 1] * s, v[:, 0] * s + v[:, 1] * c, v[:, 2]], axis=1)


def _cart_to_cyl(v: np.ndarray, ctx: Dict[str, np.ndarray]) -> np.ndarray:
    c, s = np.cos(ctx["zeta"]), np.sin(ctx["zeta"])
    return np.stack([v[:, 0] * c + v[:, 1] * s, -v[:, 0] * s + v[:, 1] * c, v[:, 2]], axis=1)


def _launch_flag_to_v4(a, ctx):
    return np.asarray("vacuum" if bool(a) else "plasma")


def _launch_flag_to_v3(a, ctx):
    return np.asarray(str(a) == "vacuum")


def _K_zeta_to_v4(a, ctx):
    # v3 stores K_zeta once (it is constant); v4 along the ray
    return np.full(ctx["tau"].shape, float(a))


def _K_zeta_to_v3(a, ctx):
    return np.asarray(a[0])


def _boundary_to_v4(a, ctx):
    return np.asarray("None" if str(a) in ("", "None") else str(a))


TAU = ("tau",)
CYL_MAT = ("row", "col")
V4_CYL_MAT = ("row_cyld", "col_cyld")
CART_MAT = ("row_cart", "col_cart")

RULES: List[Rule] = [
    # inputs
    Rule(("inputs", "Psi_BC_flag"), ("inputs", "boundary_flag"), to_v4=_boundary_to_v4),
    Rule(("inputs", "vacuumLaunch_flag"), ("inputs", "launch_flag"),
         to_v4=_launch_flag_to_v4, to_v3=_launch_flag_to_v3),
    Rule(("inputs", "quick_run"), ("inputs", "ray_tracing_flag")),
    Rule(("inputs", "interp_order"), ("inputs", "magnetic_data_interp_order")),
    Rule(("inputs", "K_initial"), ("solver_output", "K_initial_cylindrical"), ("col",), ("col_cyld",)),
    Rule(("inputs", "initial_position"), ("solver_output", "q_initial_cylindrical"), ("col",), ("col_cyld",)),
    Rule(("inputs", "Psi_3D_lab_initial"), ("solver_output", "Psi_3D_initial_labframe_cylindrical"),
         CYL_MAT, V4_CYL_MAT),
    # the ray and beam
    Rule(("solver_output", "q_R"), ("solver_output", "q_R_output")),
    Rule(("solver_output", "q_Z"), ("solver_output", "q_Z_output")),
    Rule(("solver_output", "q_zeta"), ("solver_output", "q_zeta_output")),
    Rule(("solver_output", "K_R"), ("solver_output", "K_R_output")),
    Rule(("solver_output", "K_Z"), ("solver_output", "K_Z_output")),
    Rule(("solver_output", "Psi_3D"), ("solver_output", "Psi_3D_output_labframe_cylindrical"),
         ("tau",) + CYL_MAT, ("tau",) + V4_CYL_MAT),
    Rule(("solver_output", "solver_status"), ("solver_output", "solver_status")),
    Rule(("analysis", "K_zeta_initial"), ("solver_output", "K_zeta_output"), (), TAU,
         to_v4=_K_zeta_to_v4, to_v3=_K_zeta_to_v3),
    Rule(("analysis", "K_magnitude"), ("solver_output", "K_output_magnitude")),
    Rule(("analysis", "Psi_3D_Cartesian"), ("solver_output", "Psi_3D_output_labframe_cartesian")),
    Rule(("analysis", "Psi_3D_lab_entry"), ("solver_output", "Psi_3D_entry_labframe_cylindrical"),
         CYL_MAT, V4_CYL_MAT),
    Rule(("analysis", "Psi_3D_lab_launch"), ("solver_output", "Psi_3D_launch_labframe_cylindrical"),
         CYL_MAT, V4_CYL_MAT),
    Rule(("analysis", "distance_from_launch_to_entry"), ("solver_output", "distance_from_launch_to_entry")),
    Rule(("analysis", "beam_cartesian"), ("solver_output", "q_output_cartesian"),
         ("tau", "col_cart"), ("tau", "row_cart")),
    # analysis: renamed
    Rule(("analysis", "poloidal_flux"), ("analysis", "polflux")),
    Rule(("analysis", "normalised_gyro_freqs"), ("analysis", "normalised_omega_ce")),
    Rule(("analysis", "normalised_plasma_freqs"), ("analysis", "normalised_omega_pe")),
    Rule(("analysis", "g_hat_Cartesian"), ("analysis", "g_hat_cartesian")),
    Rule(("analysis", "x_hat_Cartesian"), ("analysis", "x_hat_cartesian")),
    Rule(("analysis", "y_hat_Cartesian"), ("analysis", "y_hat_cartesian")),
    # analysis: cylindrical in v3, Cartesian in v4
    Rule(("analysis", "b_hat"), ("analysis", "b_hat_cartesian"), ("tau", "col"), ("tau", "col_cart"),
         to_v4=_cyl_to_cart, to_v3=_cart_to_cyl),
]
# analysis: same name and meaning in both
for _name in ("B_Z", "B_magnitude", "H_Booker", "H_Booker_other", "H_eigvals", "Psi_xx", "Psi_xy", "Psi_yy",
              "Psi_xg", "Psi_yg", "Psi_gg", "cutoff_index", "d_xhat_d_tau_dot_yhat", "electron_density",
              "epsilon_g", "epsilon_para", "epsilon_perp", "g_magnitude", "g_magnitude_Cardano",
              "kappa_dot_ghat", "kappa_dot_xhat", "kappa_dot_yhat", "theta_m"):
    RULES.append(Rule(("analysis", _name), ("analysis", _name)))
# analysis in v3 -> dbs_analysis in v4
for _v3, _v4 in (("M_w_inv_xx", "M_w_inv_xx"), ("M_w_inv_xy", "M_w_inv_xy"), ("M_w_inv_yy", "M_w_inv_yy"),
                 ("M_xx", "M_xx"), ("M_xy", "M_xy"), ("d_theta_d_tau", "d_theta_d_tau"),
                 ("delta_theta_m", "delta_theta_m"), ("delta_k_perp_2", "delta_k_perp2"),
                 ("k_perp_1_bs", "k_perp1_bs"), ("k_perp_1_bs_normal", "k_perp1_bs_normal"),
                 ("k_perp_1_bs_binormal", "k_perp1_bs_binormal"), ("theta", "theta"),
                 ("loc_b", "loc_b"), ("loc_m", "loc_m"), ("loc_p", "loc_p"), ("loc_r", "loc_r"),
                 ("loc_s", "loc_s_-13_3"),
                 ("xhat_dot_grad_bhat_dot_xhat", "xhat_dot_grad_bhat_dot_xhat"),
                 ("xhat_dot_grad_bhat_dot_yhat", "xhat_dot_grad_bhat_dot_yhat"),
                 ("xhat_dot_grad_bhat_dot_ghat", "xhat_dot_grad_bhat_dot_ghat"),
                 ("yhat_dot_grad_bhat_dot_xhat", "yhat_dot_grad_bhat_dot_xhat"),
                 ("yhat_dot_grad_bhat_dot_yhat", "yhat_dot_grad_bhat_dot_yhat"),
                 ("yhat_dot_grad_bhat_dot_ghat", "yhat_dot_grad_bhat_dot_ghat")):
    RULES.append(Rule(("analysis", _v3), ("dbs_analysis", _v4)))


def detect_version(dt: xr.DataTree) -> str:
    """'v3' or 'v4', from the layout"""
    if "dbs_analysis" in dt.children or "q_R_output" in dt["solver_output"].ds:
        return "v4"
    if "q_R" in dt["solver_output"].ds:
        return "v3"
    raise ValueError("Not a Scotty v3 or v4 output file")


def _context(dt: xr.DataTree, version: str) -> Dict[str, np.ndarray]:
    so = dt["solver_output"].ds
    zeta = so["q_zeta" if version == "v3" else "q_zeta_output"].values
    return {"zeta": np.asarray(zeta), "tau": np.asarray(so["tau"].values)}


def _convert(dt: xr.DataTree, source: str) -> xr.DataTree:
    target = "v4" if source == "v3" else "v3"
    ctx = _context(dt, source)
    groups: Dict[str, Dict[str, xr.DataArray]] = {}
    used = set()
    for rule in RULES:
        (sg, sn), (tg, tn) = (rule.v3, rule.v4) if source == "v3" else (rule.v4, rule.v3)
        if sg not in dt.children or sn not in dt[sg].ds.data_vars:
            continue
        src = dt[sg].ds[sn]
        transform = rule.to_v4 if source == "v3" else rule.to_v3
        dims = rule.v4_dims if source == "v3" else rule.v3_dims
        dims = src.dims if dims is None else dims
        values = np.asarray(transform(src.values, ctx))
        groups.setdefault(tg, {})[tn] = xr.DataArray(values, dims=dims, attrs=dict(src.attrs))
        used.add((sg, sn))

    # Inputs that keep their name
    other_inputs = {r.v4[1] if source == "v3" else r.v3[1] for r in RULES if r.v3[0] == "inputs"}
    for name, da in dt["inputs"].ds.data_vars.items():
        if ("inputs", name) not in used and name not in other_inputs and name in _SAME_NAME_INPUTS:
            groups.setdefault("inputs", {})[name] = da.copy()
            used.add(("inputs", name))
    # The launch position is a data variable in v3's inputs, a coordinate in v4's
    if target == "v4":
        groups.setdefault("inputs", {})["geometry"] = xr.DataArray(np.asarray("cylindrical"))
        if "launch_position" in dt["inputs"].ds:
            launch = dt["inputs"].ds["launch_position"].values
            groups["inputs"]["launch_position"] = xr.DataArray(launch, dims=("launch_position",))
            used.add(("inputs", "launch_position"))
    elif "launch_position" in dt["inputs"].ds.coords:
        groups.setdefault("inputs", {})["launch_position"] = xr.DataArray(
            dt["inputs"].ds.coords["launch_position"].values, dims=("col",)
        )

    # Coordinates along the ray
    tau = ctx["tau"]
    if source == "v3":
        arc = dt["analysis"].ds.coords.get("distance_along_line")
    else:
        arc = dt["dbs_analysis"].ds.coords.get("beam_path_length") if "dbs_analysis" in dt.children else None
    tree = {}
    for group, variables in groups.items():
        ds = xr.Dataset(variables)
        if "tau" in ds.dims:
            ds = ds.assign_coords(tau=tau)
            if arc is not None and group in ("analysis", "dbs_analysis"):
                name = "distance_along_line" if target == "v3" else "beam_path_length"
                ds = ds.assign_coords({name: ("tau", np.asarray(arc.values))})
        tree["/" + group] = ds

    not_converted = sorted(
        f"{node.path.strip('/')}/{name}"
        for node in dt.subtree
        for name in node.ds.data_vars
        if (node.path.strip("/"), name) not in used
    )
    out = xr.DataTree.from_dict(tree)
    out.attrs = dict(dt.attrs)
    out.attrs.update(
        scotty_compat_converted_from=source,
        scotty_compat_converted_to=target,
        scotty_compat_not_converted=", ".join(not_converted),
    )
    return out


# Inputs with the same name and meaning in v3 and v4
_SAME_NAME_INPUTS = {
    "atol", "rtol", "delta_K_R", "delta_K_Z", "delta_K_zeta", "delta_R", "delta_Z", "density_fit_method",
    "detailed_analysis_flag", "equil_time", "figure_flag", "find_B_method", "input_filename_suffix",
    "interp_smoothing", "launch_angular_frequency", "launch_beam_curvature", "launch_beam_width",
    "launch_freq_GHz", "len_tau", "magnetic_data_path", "mode_flag", "ne_data_path", "output_filename_suffix",
    "output_path", "poloidal_flux_enter", "poloidal_flux_zero_density", "poloidal_flux_zero_temperature",
    "poloidal_launch_angle_Torbeam", "toroidal_launch_angle_Torbeam", "shot", "relativistic_flag",
}


def to_v4_output(dt: xr.DataTree) -> xr.DataTree:
    """A v3 output tree in the v4 layout"""
    if detect_version(dt) != "v3":
        raise ValueError("Expected a v3 output tree")
    return _convert(dt, "v3")


def to_v3_output(dt: xr.DataTree) -> xr.DataTree:
    """A v4 output tree in the v3 layout"""
    if detect_version(dt) != "v4":
        raise ValueError("Expected a v4 output tree")
    return _convert(dt, "v4")


def convert_output_file(
    path_in: Union[str, pathlib.Path], path_out: Union[str, pathlib.Path]
) -> str:
    """Convert a Scotty output file to the other version's layout; returns the
    version written ('v3' or 'v4')"""
    with xr.open_datatree(path_in, engine="h5netcdf") as dt:
        dt = dt.load()
    version = detect_version(dt)
    out = to_v4_output(dt) if version == "v3" else to_v3_output(dt)
    out.to_netcdf(path_out, engine="h5netcdf", invalid_netcdf=True)
    return out.attrs["scotty_compat_converted_to"]
