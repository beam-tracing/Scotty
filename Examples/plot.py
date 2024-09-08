# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 12:43:55 2023

@author: VH Hall-Chen

Incorporate stuff from the old outplot.py into this

TODO: Different options for plotting.
- Plot individual graphs
- Plot groups 'all', 'basic', 'advanced', 'troubleshooting'
"""

import datatree
from scotty.plotting import (
    plot_dispersion_relation,
    plot_poloidal_beam_path,
    plot_toroidal_beam_path
    )

path = 'D:\\Dropbox\\VHChen2022\\Code - Testing Scotty\\Various cases\\DIII-D - DBS240\\Output\\'
dt = datatree.open_datatree(path+"scotty_output.h5", engine="h5netcdf")


# plot_dispersion_relation(dt['analysis'])

plot_poloidal_beam_path(dt,zoom=True)

# plot_toroidal_beam_path(dt)

# dt.close()