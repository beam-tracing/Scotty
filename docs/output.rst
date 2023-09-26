.. _output:

Output parameters
=================

:func:`beam_me_up` returns a `datatree`_ instance which contains the
simulation inputs under ``inputs``, the immediate results under
``solver_output``, and some further processed results under
``analysis``.

Datatrees are collections of `xarray`_ ``Datasets``, which themselves
are collections of N-dimensional variables with labelled coordinates.

See `beam_me_up() <scotty.beam_me_up.beam_me_up>` for a detailed
description of input parameters.

All the outputs are in cylindrical coordinates unless otherwise stated
(``Cartesian`` or with the lower case ``x`` or ``y``).

The ``solver_output`` group in the output datatree contains:

* ``q_R`` - beam trajectory, :math:`R` component
* ``q_zeta`` - beam trajectory, :math:`\zeta` component
* ``q_Z`` - beam trajectory, :math:`Z` component
* ``K_R`` - :math:`R` component of the wave vector
* ``K_Z`` - :math:`Z` component of the wave vector
* ``Psi_3D`` - wave tensor

These variables are all functions of the beam parameter :math:`\tau`.

The ``analysis`` group in the output datatree contains lots of derived
parameters and quantities. Some of these are:

* ``Psi_xx`` - :math:`xx` component of :math:`\Psi_w`
* ``Psi_xy`` - :math:`xy` component of :math:`\Psi_w`
* ``Psi_yy`` - :math:`yy` component of :math:`\Psi_w`

Output files
============

Data is saved to netCDF files using `xarray`_ and the `h5netcdf`_
engine. This allows us to easily save things like :math:`\Psi` which
is complex. At the time of writing (2023), ``h5netcdf`` writes complex
numbers using a method which is incompatible which current releases of
netCDF (although this has been fixed and should be available in the
next netCDF release). For this reason, the default filename is
``scotty_output.h5``.

The best way to read these files with `datatree`_, specifying the
``h5netcdf`` engine::

  import datatree

  dt = datatree.open_datatree("scotty_output.h5", engine="h5netcdf")


.. _xarray: https://xarray.pydata.org
.. _h5netcdf: https://h5netcdf.org
.. _datatree: https://xarray-datatree.readthedocs.io/en/latest/
