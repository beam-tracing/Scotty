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
- ``quick_run`` to ``ray_tracing_flag``;
- ``output_path`` given as a string, to a ``pathlib.Path``;
- defaults that changed between v3 and v4, restored when the v3 script left
  them unset: ``interp_smoothing`` (0), ``find_B_method`` ("torbeam"),
  ``detailed_analysis_flag`` (True), and ``poloidal_flux_zero_density`` and
  ``poloidal_flux_zero_temperature`` (1.01, raised to
  ``poloidal_flux_enter`` if that is larger, since v4 requires it).

``vacuum_propagation_flag`` and ``resonance_flag`` have no v4 equivalent and
are dropped with a warning.
"""

import logging
import pathlib
from typing import Any, Dict

import numpy as np

from scotty.fun_general import find_K_labframe_cyl_to_cart

log = logging.getLogger(__name__)

#: v3 arguments that v4's beam_me_up no longer takes
V3_ONLY_ARGUMENTS = (
    "vacuumLaunch_flag",
    "vacuum_propagation_flag",
    "Psi_BC_flag",
    "resonance_flag",
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

    for name in ("vacuum_propagation_flag", "resonance_flag"):
        if name in kw:
            _warn(f"dropped `{name}` = {kw.pop(name)!r}: it has no v4 equivalent")

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
