# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 12:43:55 2023

@author: VH Hall-Chen

Plots of a Scotty v4 output: the dispersion relation along the beam, and the
beam path in the poloidal and toroidal planes.

Run with the path of an output file (``scotty_output*.h5``) to plot it, or
with no argument to run the built-in ``DBS_synthetic`` case first::

    python plot.py [path/to/scotty_output.h5]

The flux surfaces are drawn only when the field is available (here, for the
built-in case), since v4's output does not store the poloidal flux on a grid.

TODO: Different options for plotting.
- Plot individual graphs
- Plot groups 'all', 'basic', 'advanced', 'troubleshooting'
"""

import sys
import tempfile

import matplotlib.pyplot as plt
import xarray as xr

from scotty import beam_me_up, get_parameters_for_Scotty
from scotty.plotting import (
    plot_dispersion_relation,
    plot_poloidal_beam_path,
    plot_toroidal_beam_path,
)

if len(sys.argv) > 1:
    dt = xr.open_datatree(sys.argv[1], engine="h5netcdf")
    field = None
else:
    kwargs = get_parameters_for_Scotty("DBS_synthetic")
    dt, field = beam_me_up(
        **kwargs, output_path=tempfile.mkdtemp(), return_dt_field=True
    )

plot_dispersion_relation(dt["analysis"])
plot_poloidal_beam_path(dt, zoom=True, field=field)
plot_toroidal_beam_path(dt, field=field)

plt.show()
