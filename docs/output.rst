.. _output:

Output
======

`beam_me_up() <scotty.beam_me_up.beam_me_up>` returns an
`xarray.DataTree`, a tree of `xarray.Dataset` objects with labelled
coordinates, and saves it to
``output_path / f"scotty_output{output_filename_suffix}.h5"``. It has four
groups:

``inputs``
    The inputs of the run, as checked, with the grid of the magnetic field
    (coordinates ``R`` and ``Z``, or ``X``, ``Y`` and ``Z``). The data paths
    are those of the files that the run read (``"None"`` for none), and
    ``magnetic_data_time`` is the time of a field that a constructor read
    from a file with several times (NaN otherwise).

``solver``
    What the solvers return, along the beam parameter :math:`\tau`: the
    position ``q_output_*``, the wavevector ``K_output_*`` and
    :math:`\Psi` (``Psi_3D_output_labframe_*``), in both coordinate systems
    (``_cartesian`` and ``_cylindrical``), and their values at the launch,
    at the plasma boundary before the boundary conditions, and after them
    where the solvers start (``*_launch``, ``*_entry``, ``*_initial``; see
    :ref:`naming`). In cylindrical coordinates ``K_zeta`` is the
    toroidal mode number, not the toroidal component of :math:`K`.

``beam``
    The beam along the ray (`scotty.analysis.basic_analysis`): the beam
    frame, :math:`\Psi_w` and its components (``Psi_xx``, ``Psi_xy``,
    ``Psi_yy``), the widths and curvatures along their principal axes
    (``width_1``, ``curvature_1``, ...), the dielectric tensor, the
    distances along the beam (``beam_path_length``) and the consistency
    checks (``H_Booker``, ``H_Cardano``, ``Psi_g_residual_*``). With
    ``analysis_flag=False`` there is no ``beam`` group.

``diagnostics/dbs``
    Doppler backscattering (`scotty.diagnostics.dbs.dbs_analysis`): the
    backscattered wavenumbers (``k_perp1_bs``, ``k_perp2_bs``), the filter
    functions (``filterfun_*``) and their integrals along the beam, and a
    model turbulence spectrum (``delta_ne_sq_*``). With
    ``dbs_analysis_flag=False`` there is no ``diagnostics/dbs`` group, and
    the variables of ``beam`` that the DBS analysis computes
    (`scotty.data_io.BEAM_FROM_DBS`) are left out too.

See :ref:`code-layout` for what goes in each group, and
`beam_me_up() <scotty.beam_me_up.beam_me_up>` for the units and the names
of the coordinates.

Output files
------------

The output is saved as netCDF with `xarray`_ and the `h5netcdf`_ engine,
with ``invalid_netcdf=True`` so that complex numbers such as :math:`\Psi`
can be stored. Read it with `xarray.open_datatree`, with the same engine::

    import xarray

    dt = xarray.open_datatree("scotty_output.h5", engine="h5netcdf")

`scotty.compat.convert_output_file` converts an output file between the
layouts of Scotty v3 and v4.

.. _xarray: https://xarray.pydata.org
.. _h5netcdf: https://h5netcdf.org
