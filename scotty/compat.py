# -*- coding: utf-8 -*-
# Copyright 2018 - 2023, Valerian Hall-Chen and the Scotty contributors
# SPDX-License-Identifier: GPL-3.0

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

- the launch flags: ``vacuumLaunch_flag`` and ``vacuum_propagation_flag`` to
  ``launch_flag``. ``vacuumLaunch_flag=False`` is ``"plasma"``.
  ``vacuumLaunch_flag=True`` (v3's default) is ``"vacuum_analytic"`` if
  ``vacuum_propagation_flag`` is True, and ``"vacuum_numerical"`` otherwise
  (v3's ``beam_me_up`` defaulted to ``vacuum_propagation_flag=False``, so a
  script that does not set it gets ``"vacuum_numerical"``). v3's docstring
  describes ``vacuum_propagation_flag`` the wrong way round: its code (and
  this conversion) propagates analytically through the vacuum when it is
  True;
- ``Psi_BC_flag`` is passed through unchanged (v4 uses the same name and the
  same values; v3's deprecated booleans raise in v4). If it is not given,
  v3's default ``None`` is set. With ``"vacuum_numerical"`` it is set to
  ``None``, with a warning: v3 propagated through the vacuum numerically
  without boundary conditions and ignored the flag, and v4 refuses it there;
- ``geometry`` is set to ``"cylindrical"``, the only geometry v3 had;
- a plasma launch: ``plasmaLaunch_K`` (cylindrical, as in v3) to the
  Cartesian ``plasmaLaunch_K_cartesian``, and
  ``plasmaLaunch_Psi_3D_lab_Cartesian`` to
  ``plasmaLaunch_Psi_3D_lab_cartesian``;
- ``quick_run`` to ``ray_tracing_flag``; ``quick_cutoff(dt)`` then gives the
  numbers v3's ``quick_run`` returned (R, Z, |K|, poloidal flux and θ_m at the
  minimum of |K|);
- ``output_path`` given as a string, to a ``pathlib.Path`` (v4 accepts
  either, but v3 used a string as a file-name prefix, which is warned about);
- defaults that differ between v3 and v4, restored when the v3 script left
  them unset: ``poloidal_flux_zero_density`` and
  ``poloidal_flux_zero_temperature`` (1.01, raised to
  ``poloidal_flux_enter`` if that is larger, since v4 requires it); the
  spatial finite-difference steps ``delta_R`` = -1e-4 and ``delta_Z`` = 1e-4
  (v4's default is 1e-5 for both; v3 had no other spatial step); the solver
  tolerances ``rtol`` = 1e-3 and ``atol`` = 1e-6 (v4's are 1e-4, 1e-7); and the
  number of output points ``len_tau`` = 102 (v4's is 4002).

``resonance_flag``, a bool in v3, becomes v4's integer: ``True`` is 2 (stop at the
fundamental and the second harmonic) and ``False`` is 0. v3's ``False`` let the ray
carry on past the fundamental; v4 raises there instead, as the cold-plasma
Hamiltonian is singular and the rest of the ray would be wrong (#169).

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

from scotty.data_io import from_output_tree, to_output_tree
from scotty.fun_general import find_K_labframe_cyl_to_cart

log = logging.getLogger(__name__)

#: v3 arguments that v4's beam_me_up no longer takes
V3_ONLY_ARGUMENTS = (
    "vacuumLaunch_flag",
    "vacuum_propagation_flag",
    "quick_run",
    "plasmaLaunch_K",
    "plasmaLaunch_Psi_3D_lab_Cartesian",
)


V3_DEFAULT_POLOIDAL_FLUX_ZERO = 1.01

#: v3's beam_me_up defaults for the spatial finite-difference steps
#: (upstream/master feb9499); v4's are 1e-5
V3_DEFAULT_SPATIAL_STEPS = {"delta_R": -1e-4, "delta_Z": 1e-4}

#: v3's beam_me_up defaults for the solver tolerances; v4's are 1e-4, 1e-7
V3_DEFAULT_TOLERANCES = {"rtol": 1e-3, "atol": 1e-6}

#: v3's beam_me_up default for the number of output points; v4's is 4002
V3_DEFAULT_LEN_TAU = 102


_warned = set()


def _warn(message: str) -> None:
    # Once per message, so that a sweep calling this for every run stays readable
    if message not in _warned:
        _warned.add(message)
        log.warning(f"from_v3_kwargs: {message}")


def from_v3_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of ``kwargs`` with v3 arguments converted to v4"""
    kw = dict(kwargs)

    # Launch from vacuum (propagated analytically or numerically) or from
    # inside the plasma. v3's defaults: vacuumLaunch_flag = True,
    # vacuum_propagation_flag = False
    vacuum_launch_given = "vacuumLaunch_flag" in kw
    propagation_given = "vacuum_propagation_flag" in kw
    vacuum_launch = bool(kw.pop("vacuumLaunch_flag", True))
    vacuum_propagation = bool(kw.pop("vacuum_propagation_flag", False))
    if "launch_flag" not in kw:
        if not vacuum_launch:
            kw["launch_flag"] = "plasma"
        elif vacuum_propagation:
            kw["launch_flag"] = "vacuum_analytic"
        else:
            kw["launch_flag"] = "vacuum_numerical"

        def given(name: str, value: bool, was_given: bool) -> str:
            return f"`{name}` = {value}" + ("" if was_given else " (v3's default)")

        source = given("vacuumLaunch_flag", vacuum_launch, vacuum_launch_given)
        if vacuum_launch:
            source += ", " + given(
                "vacuum_propagation_flag", vacuum_propagation, propagation_given
            )
        _warn(f"{source} -> `launch_flag` = '{kw['launch_flag']}'")

    # Plasma-vacuum boundary conditions: same name and values in v4
    if "Psi_BC_flag" not in kw:
        kw["Psi_BC_flag"] = None  # v3's default
        _warn("set `Psi_BC_flag` = None (v3's default)")
    elif kw["launch_flag"] == "vacuum_numerical" and kw["Psi_BC_flag"] is not None:
        # v3 ignored it when propagating numerically through the vacuum (no
        # boundary conditions are applied there); v4 refuses it
        _warn(
            f"dropped `Psi_BC_flag` = {kw['Psi_BC_flag']!r}: v3 ignored it with "
            "`launch_flag` = 'vacuum_numerical' (no boundary conditions), so "
            "`Psi_BC_flag` = None"
        )
        kw["Psi_BC_flag"] = None

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
        if "plasmaLaunch_Psi_3D_lab_cartesian" not in kw and np.any(
            np.asarray(Psi) != 0
        ):
            kw["plasmaLaunch_Psi_3D_lab_cartesian"] = Psi
            _warn(
                "`plasmaLaunch_Psi_3D_lab_Cartesian` -> `plasmaLaunch_Psi_3D_lab_cartesian`"
            )

    if "quick_run" in kw:
        quick_run = bool(kw.pop("quick_run"))
        if "ray_tracing_flag" not in kw:
            kw["ray_tracing_flag"] = quick_run
            _warn(f"`quick_run` -> `ray_tracing_flag` = {quick_run}")

    # v3's bool -> v4's integer. v3's False carried on past the fundamental, which
    # v4 refuses (#169), so say so
    if isinstance(kw.get("resonance_flag"), (bool, np.bool_)):
        resonance_flag = bool(kw["resonance_flag"])
        kw["resonance_flag"] = 2 if resonance_flag else 0
        _warn(
            f"`resonance_flag` = {resonance_flag} -> {kw['resonance_flag']}"
            + (
                ""
                if resonance_flag
                else " (v4 raises at the fundamental, where v3 carried on)"
            )
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

    enter = kw.get("poloidal_flux_enter", 1.0)
    for name in ("poloidal_flux_zero_density", "poloidal_flux_zero_temperature"):
        if name not in kw:
            kw[name] = max(V3_DEFAULT_POLOIDAL_FLUX_ZERO, enter)
            _warn(
                f"set `{name}` = {kw[name]} (v3's default, at least `poloidal_flux_enter`)"
            )

    for name, value in V3_DEFAULT_SPATIAL_STEPS.items():
        if name not in kw:
            kw[name] = value
            _warn(f"set `{name}` = {value} (v3's default; v4's is 1e-5)")

    for name, value in V3_DEFAULT_TOLERANCES.items():
        if name not in kw:
            kw[name] = value
            _warn(f"set `{name}` = {value} (v3's default; v4's are 1e-4, 1e-7)")

    if "len_tau" not in kw:
        kw["len_tau"] = V3_DEFAULT_LEN_TAU
        _warn(f"set `len_tau` = {V3_DEFAULT_LEN_TAU} (v3's default; v4's is 4002)")

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
    so, an = dt["solver"].ds, dt["beam"].ds
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
    # "v3" or "v4": apply only when converting from that version
    only_from: Optional[str] = None


# Lab-frame vectors along the ray, (tau, 3): cylindrical (R, zeta, Z) physical
# components <-> Cartesian (X, Y, Z), at the ray's toroidal angle zeta
def _cyl_to_cart(v: np.ndarray, ctx: Dict[str, np.ndarray]) -> np.ndarray:
    c, s = np.cos(ctx["zeta"]), np.sin(ctx["zeta"])
    return np.stack(
        [v[:, 0] * c - v[:, 1] * s, v[:, 0] * s + v[:, 1] * c, v[:, 2]], axis=1
    )


def _cart_to_cyl(v: np.ndarray, ctx: Dict[str, np.ndarray]) -> np.ndarray:
    c, s = np.cos(ctx["zeta"]), np.sin(ctx["zeta"])
    return np.stack(
        [v[:, 0] * c + v[:, 1] * s, -v[:, 0] * s + v[:, 1] * c, v[:, 2]], axis=1
    )


def _launch_flag_to_v4(a, ctx):
    # v3's vacuumLaunch_flag, with its vacuum_propagation_flag in the context
    if not bool(a):
        return np.asarray("plasma")
    return np.asarray(
        "vacuum_analytic" if ctx["vacuum_propagation_flag"] else "vacuum_numerical"
    )


def _launch_flag_to_v3(a, ctx):
    # Any vacuum launch is v3's vacuumLaunch_flag = True ("vacuum" is how v4
    # files before 4.0.0.dev0 stored "vacuum_analytic")
    return np.asarray(str(a) != "plasma")


def _vacuum_propagation_flag_to_v3(a, ctx):
    return np.asarray(str(a) in ("vacuum_analytic", "vacuum"))


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
    Rule(("inputs", "Psi_BC_flag"), ("inputs", "Psi_BC_flag"), to_v4=_boundary_to_v4),
    # v4 files before 4.0.0.dev0 called it `boundary_flag`
    Rule(("inputs", "Psi_BC_flag"), ("inputs", "boundary_flag"), only_from="v4"),
    Rule(
        ("inputs", "vacuumLaunch_flag"),
        ("inputs", "launch_flag"),
        to_v4=_launch_flag_to_v4,
        to_v3=_launch_flag_to_v3,
    ),
    # v4's launch_flag also gives v3's vacuum_propagation_flag; from v3, the
    # flag is read through the context, by the rule above
    Rule(
        ("inputs", "vacuum_propagation_flag"),
        ("inputs", "launch_flag"),
        to_v3=_vacuum_propagation_flag_to_v3,
        only_from="v4",
    ),
    Rule(("inputs", "quick_run"), ("inputs", "ray_tracing_flag")),
    Rule(("inputs", "interp_order"), ("inputs", "magnetic_data_interp_order")),
    Rule(
        ("inputs", "K_initial"),
        ("solver_output", "K_initial_cylindrical"),
        ("col",),
        ("col_cyld",),
    ),
    Rule(
        ("inputs", "initial_position"),
        ("solver_output", "q_initial_cylindrical"),
        ("col",),
        ("col_cyld",),
    ),
    Rule(
        ("inputs", "Psi_3D_lab_initial"),
        ("solver_output", "Psi_3D_initial_labframe_cylindrical"),
        CYL_MAT,
        V4_CYL_MAT,
    ),
    # the ray and beam
    Rule(("solver_output", "q_R"), ("solver_output", "q_R_output")),
    Rule(("solver_output", "q_Z"), ("solver_output", "q_Z_output")),
    Rule(("solver_output", "q_zeta"), ("solver_output", "q_zeta_output")),
    Rule(("solver_output", "K_R"), ("solver_output", "K_R_output")),
    Rule(("solver_output", "K_Z"), ("solver_output", "K_Z_output")),
    Rule(
        ("solver_output", "Psi_3D"),
        ("solver_output", "Psi_3D_output_labframe_cylindrical"),
        ("tau",) + CYL_MAT,
        ("tau",) + V4_CYL_MAT,
    ),
    Rule(("solver_output", "solver_status"), ("solver_output", "solver_status")),
    Rule(
        ("analysis", "K_zeta_initial"),
        ("solver_output", "K_zeta_output"),
        (),
        TAU,
        to_v4=_K_zeta_to_v4,
        to_v3=_K_zeta_to_v3,
    ),
    Rule(("analysis", "K_magnitude"), ("solver_output", "K_output_magnitude")),
    Rule(
        ("analysis", "Psi_3D_Cartesian"),
        ("solver_output", "Psi_3D_output_labframe_cartesian"),
    ),
    Rule(
        ("analysis", "Psi_3D_lab_entry"),
        ("solver_output", "Psi_3D_entry_labframe_cylindrical"),
        CYL_MAT,
        V4_CYL_MAT,
    ),
    Rule(
        ("analysis", "Psi_3D_lab_launch"),
        ("solver_output", "Psi_3D_launch_labframe_cylindrical"),
        CYL_MAT,
        V4_CYL_MAT,
    ),
    Rule(
        ("analysis", "distance_from_launch_to_entry"),
        ("solver_output", "distance_from_launch_to_entry"),
    ),
    Rule(
        ("analysis", "beam_cartesian"),
        ("solver_output", "q_output_cartesian"),
        ("tau", "col_cart"),
        ("tau", "row_cart"),
    ),
    # analysis: renamed
    Rule(("analysis", "poloidal_flux"), ("analysis", "polflux")),
    Rule(("analysis", "normalised_gyro_freqs"), ("analysis", "normalised_omega_ce")),
    Rule(("analysis", "normalised_plasma_freqs"), ("analysis", "normalised_omega_pe")),
    Rule(("analysis", "g_hat_Cartesian"), ("analysis", "g_hat_cartesian")),
    Rule(("analysis", "x_hat_Cartesian"), ("analysis", "x_hat_cartesian")),
    Rule(("analysis", "y_hat_Cartesian"), ("analysis", "y_hat_cartesian")),
    # analysis: cylindrical in v3, Cartesian in v4
    Rule(
        ("analysis", "b_hat"),
        ("analysis", "b_hat_cartesian"),
        ("tau", "col"),
        ("tau", "col_cart"),
        to_v4=_cyl_to_cart,
        to_v3=_cart_to_cyl,
    ),
]
# analysis: same name and meaning in both
for _name in (
    "B_Z",
    "B_magnitude",
    "H_Booker",
    "H_Booker_other",
    "H_eigvals",
    "Psi_xx",
    "Psi_xy",
    "Psi_yy",
    "Psi_xg",
    "Psi_yg",
    "Psi_gg",
    "cutoff_index",
    "d_xhat_d_tau_dot_yhat",
    "electron_density",
    "epsilon_g",
    "epsilon_para",
    "epsilon_perp",
    "g_magnitude",
    "g_magnitude_Cardano",
    "kappa_dot_ghat",
    "kappa_dot_xhat",
    "kappa_dot_yhat",
    "theta_m",
):
    RULES.append(Rule(("analysis", _name), ("analysis", _name)))
# analysis in v3 -> dbs_analysis in v4
for _v3, _v4 in (
    ("M_w_inv_xx", "M_w_inv_xx"),
    ("M_w_inv_xy", "M_w_inv_xy"),
    ("M_w_inv_yy", "M_w_inv_yy"),
    ("M_xx", "M_xx"),
    ("M_xy", "M_xy"),
    ("d_theta_d_tau", "d_theta_d_tau"),
    ("delta_theta_m", "delta_theta_m"),
    ("delta_k_perp_2", "delta_k_perp2"),
    ("k_perp_1_bs", "k_perp1_bs"),
    ("k_perp_1_bs_normal", "k_perp1_bs_normal"),
    ("k_perp_1_bs_binormal", "k_perp1_bs_binormal"),
    ("theta", "theta"),
    ("loc_b", "filterfun_b"),
    ("loc_m", "filterfun_m"),
    ("loc_p", "filterfun_p"),
    ("loc_r", "filterfun_r"),
    ("loc_s", "delta_ne_sq_-13_3"),
    ("xhat_dot_grad_bhat_dot_xhat", "xhat_dot_grad_bhat_dot_xhat"),
    ("xhat_dot_grad_bhat_dot_yhat", "xhat_dot_grad_bhat_dot_yhat"),
    ("xhat_dot_grad_bhat_dot_ghat", "xhat_dot_grad_bhat_dot_ghat"),
    ("yhat_dot_grad_bhat_dot_xhat", "yhat_dot_grad_bhat_dot_xhat"),
    ("yhat_dot_grad_bhat_dot_yhat", "yhat_dot_grad_bhat_dot_yhat"),
    ("yhat_dot_grad_bhat_dot_ghat", "yhat_dot_grad_bhat_dot_ghat"),
):
    RULES.append(Rule(("analysis", _v3), ("dbs_analysis", _v4)))


def detect_version(dt: xr.DataTree) -> str:
    """'v3' or 'v4', from the layout"""
    # "dbs_analysis": files from v4.0.0.dev0, before the output had its groups
    if "beam" in dt.children or "dbs_analysis" in dt.children:
        return "v4"
    if "q_R_output" in dt["solver_output"].ds:
        return "v4"
    if "q_R" in dt["solver_output"].ds:
        return "v3"
    raise ValueError("Not a Scotty v3 or v4 output file")


def _context(dt: xr.DataTree, version: str) -> Dict[str, np.ndarray]:
    so = dt["solver_output"].ds
    zeta = so["q_zeta" if version == "v3" else "q_zeta_output"].values
    context = {"zeta": np.asarray(zeta), "tau": np.asarray(so["tau"].values)}
    if version == "v3":
        inputs = dt["inputs"].ds
        # v3's beam_me_up default, for files that do not store it
        context["vacuum_propagation_flag"] = bool(
            inputs["vacuum_propagation_flag"].values
            if "vacuum_propagation_flag" in inputs
            else False
        )
    return context


def _convert(dt: xr.DataTree, source: str) -> xr.DataTree:
    target = "v4" if source == "v3" else "v3"
    ctx = _context(dt, source)
    groups: Dict[str, Dict[str, xr.DataArray]] = {}
    used = set()
    for rule in RULES:
        if rule.only_from not in (None, source):
            continue
        (sg, sn), (tg, tn) = (
            (rule.v3, rule.v4) if source == "v3" else (rule.v4, rule.v3)
        )
        if sg not in dt.children or sn not in dt[sg].ds.data_vars:
            continue
        src = dt[sg].ds[sn]
        transform = rule.to_v4 if source == "v3" else rule.to_v3
        dims = rule.v4_dims if source == "v3" else rule.v3_dims
        dims = src.dims if dims is None else dims
        values = np.asarray(transform(src.values, ctx))
        groups.setdefault(tg, {})[tn] = xr.DataArray(
            values, dims=dims, attrs=dict(src.attrs)
        )
        used.add((sg, sn))

    # Inputs that keep their name
    other_inputs = {
        r.v4[1] if source == "v3" else r.v3[1]
        for r in RULES
        if r.v3[0] == "inputs" and r.only_from in (None, source)
    }
    # Consumed through the context (to make v4's launch_flag)
    if source == "v3" and "vacuum_propagation_flag" in dt["inputs"].ds:
        used.add(("inputs", "vacuum_propagation_flag"))
    for name, da in dt["inputs"].ds.data_vars.items():
        if (
            ("inputs", name) not in used
            and name not in other_inputs
            and name in _SAME_NAME_INPUTS
        ):
            groups.setdefault("inputs", {})[name] = da.copy()
            used.add(("inputs", name))
    # The launch position is a data variable in v3's inputs, a coordinate in v4's
    if target == "v4":
        groups.setdefault("inputs", {})["geometry"] = xr.DataArray(
            np.asarray("cylindrical")
        )
        if "launch_position" in dt["inputs"].ds:
            launch = dt["inputs"].ds["launch_position"].values
            groups["inputs"]["launch_position"] = xr.DataArray(
                launch, dims=("launch_position",)
            )
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
        arc = (
            dt["dbs_analysis"].ds.coords.get("beam_path_length")
            if "dbs_analysis" in dt.children
            else None
        )
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
    "atol",
    "rtol",
    "delta_K_R",
    "delta_K_Z",
    "delta_K_zeta",
    "delta_R",
    "delta_Z",
    "density_fit_method",
    "detailed_analysis_flag",
    "equil_time",
    "figure_flag",
    "find_B_method",
    "input_filename_suffix",
    "interp_smoothing",
    "launch_angular_frequency",
    "launch_beam_curvature",
    "launch_beam_width",
    "launch_freq_GHz",
    "len_tau",
    "magnetic_data_path",
    "mode_flag",
    "ne_data_path",
    "output_filename_suffix",
    "output_path",
    "poloidal_flux_enter",
    "poloidal_flux_zero_density",
    "poloidal_flux_zero_temperature",
    "poloidal_launch_angle_Torbeam",
    "toroidal_launch_angle_Torbeam",
    "shot",
    "relativistic_flag",
}


def to_v4_output(dt: xr.DataTree) -> xr.DataTree:
    """A v3 output tree in the v4 layout"""
    if detect_version(dt) != "v3":
        raise ValueError("Expected a v3 output tree")
    out = _convert(dt, "v3")
    flat = {name: out[name].to_dataset() for name in out.children}
    empty = xr.Dataset()
    tree = to_output_tree(
        flat.get("inputs", empty),
        flat.get("solver_output", empty),
        flat.get("analysis", empty),
        flat.get("dbs_analysis", empty),
    )
    tree.attrs = out.attrs
    return tree


def to_v3_output(dt: xr.DataTree) -> xr.DataTree:
    """A v4 output tree in the v3 layout"""
    if detect_version(dt) != "v4":
        raise ValueError("Expected a v4 output tree")
    if "beam" in dt.children:
        attrs = dt.attrs
        dt = xr.DataTree.from_dict(from_output_tree(dt))
        dt.attrs = attrs
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
