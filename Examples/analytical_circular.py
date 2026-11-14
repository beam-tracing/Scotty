"""A DBS beam in an analytical field with circular flux surfaces.

The field has circular flux surfaces (normalised poloidal flux
psi = (r/a)^2) and the density is quadratic in psi, zero at psi = 1. The beam
is launched in vacuum and propagated analytically to the plasma, where the
continuous boundary condition is applied: the density is zero there, but its
gradient is not.

Run it as::

    python analytical_circular.py [output_folder]

and plot the result with ``plot.py``.

The second part runs the same beam into a density that jumps at the plasma
edge, which needs the discontinuous boundary condition, and a short scan in
the toroidal launch angle.
"""

import pathlib
import sys

import numpy as np

from scotty import beam_me_up
from scotty.profile_fit import PolynomialFit, QuadraticFit

output_path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "scotty_output")
output_path.mkdir(exist_ok=True)

beam = dict(
    geometry="cylindrical",
    # Launch: TORBEAM angles (degrees) from the antenna, at (R, zeta, Z) in m
    poloidal_launch_angle_Torbeam=6.0,
    toroidal_launch_angle_Torbeam=0.0,
    launch_position=np.array([2.587, 0.0, -0.0157]),
    launch_freq_GHz=55.0,
    mode_flag=1,  # O-mode
    launch_beam_width=0.04,  # m
    launch_beam_curvature=-0.25,  # 1/m: a beam focused 4 m ahead
    # Analytical field: circular flux surfaces
    find_B_method="analytical",
    B_T_axis=1.0,  # T
    R_axis=1.5,  # m
    minor_radius_a=0.5,  # m
    B_p_a=0.1,  # T
    # Vacuum propagation to the plasma, then the boundary condition. The plasma
    # starts where the density profile ends, its zero point (here psi = 1)
    launch_flag="vacuum_analytic",
)

# 1. Density 4e19 m^-3 at the axis, zero at psi = 1 (the first argument)
dt = beam_me_up(
    **beam,
    density_fit_method=QuadraticFit(1.0, 4.0),
    Psi_BC_flag="continuous",
    output_path=output_path,
    output_filename_suffix="_continuous",
)
solver = dt["solver"]
cutoff = int(solver["K_output_magnitude"].argmin())
print(
    f"Cut-off (minimum |K|) at R = {float(solver['q_R_output'][cutoff]):.4f} m, "
    f"Z = {float(solver['q_Z_output'][cutoff]):.4f} m"
)

# 2. A density that jumps at the edge: 4 - (4/1.3) psi^2, cut at psi = 1,
#    where it is still about 0.9e19 m^-3. The continuous condition would
#    refuse it; the discontinuous one applies the jump
beam_me_up(
    **beam,
    density_fit_method=PolynomialFit(1.0, -4.0 / 1.3, 0.0, 4.0),
    Psi_BC_flag="discontinuous",
    output_path=output_path,
    output_filename_suffix="_discontinuous",
)

# 3. A scan in the toroidal launch angle, one output file per angle
for toroidal_angle in (-4.0, -2.0, 0.0, 2.0, 4.0):
    beam_me_up(
        **dict(beam, toroidal_launch_angle_Torbeam=toroidal_angle),
        density_fit_method=QuadraticFit(1.0, 4.0),
        Psi_BC_flag="continuous",
        output_path=output_path,
        output_filename_suffix=f"_tor{toroidal_angle:+.0f}",
    )
