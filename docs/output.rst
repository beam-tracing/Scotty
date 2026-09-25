.. _output:

Reading and interpreting results
================================

:func:`scotty.beam_me_up.beam_me_up` returns an ``xarray.DataTree`` and saves
the same tree to an HDF5 file. The tree groups related ``xarray.Dataset``
objects together; each variable has named dimensions and coordinates, which
makes it possible to select and plot parts of the result without guessing
array indices.

Explore a result
----------------

Open a saved result with the ``h5netcdf`` engine:

.. code-block:: python

   import xarray as xr

   tree = xr.open_datatree("results/scotty_output_example.h5", engine="h5netcdf")
   print(tree)

The main groups are:

``inputs``
   The settings and input grids used for the run, including launch
   configuration, input paths, and magnetic geometry.
``solver_output``
   The beam trajectory, wavevector components, and propagated complex-valued
   ``Psi_3D`` wave/beam tensor.
``analysis``
   Derived quantities evaluated along the trajectory, such as magnetic field,
   density, poloidal flux, beam geometry, and localization.

For example, inspect the ray and density along it:

.. code-block:: python

   ray = tree.solver_output
   analysis = tree.analysis

   print(ray.q_R)                   # major radius, in metres
   print(ray.q_zeta)                # toroidal angle, in radians
   print(ray.q_Z)                   # vertical position, in metres
   print(analysis.electron_density) # density along the ray

The core coordinates are cylindrical: ``q_R`` is major radius, ``q_zeta`` is
toroidal angle, and ``q_Z`` is height. Cartesian trajectory coordinates are
also available in ``analysis.beam_cartesian``. The solver's independent
coordinate ``tau`` parametrizes integration along the beam; it is not the
physical distance travelled. Use ``analysis.distance_along_line`` for distance
from the launch point, or ``analysis.l_lc`` for signed distance from the
analysis cutoff location.

Find the analysis cutoff point and inspect the corresponding ray location:

.. code-block:: python

   cutoff_index = int(analysis.cutoff_index.item())
   cutoff_R = ray.q_R.isel(tau=cutoff_index).item()
   cutoff_Z = ray.q_Z.isel(tau=cutoff_index).item()
   print(f"Cutoff estimate: R={cutoff_R:.3f} m, Z={cutoff_Z:.3f} m")

Scotty identifies this point from the minimum wavevector magnitude along the
computed trajectory. Treat it as an estimate from that run, and check that the
trajectory and equilibrium cover the region you intend to study.

Useful variables
----------------

``solver_output`` contains the direct solver quantities. The trajectory
``q_R``, ``q_zeta``, and ``q_Z``, the wavevector components ``K_R`` and ``K_Z``,
and ``Psi_3D`` are sampled along ``tau``. ``Psi_3D`` is a complex-valued tensor;
its ``row`` and ``col`` coordinates label cylindrical components.

``analysis`` contains quantities that help check and interpret a run:

* ``poloidal_flux`` and ``electron_density`` show where the ray travels
  relative to the plasma profile.
* ``K_magnitude`` and ``cutoff_index`` locate Scotty's cutoff estimate;
  ``theta`` and ``theta_m`` describe beam/magnetic-field and wavevector/field
  angles, respectively.
* ``Psi_xx``, ``Psi_xy``, and ``Psi_yy`` are beam-tensor components in the
  transverse beam basis. Their imaginary parts are related to beam width;
  they are generally complex and should not be treated as ordinary positions.
* ``loc_b``, ``loc_r``, ``loc_s``, ``loc_m``, and ``loc_p`` are factors in the
  localization model. ``loc_b_r_s`` and ``loc_b_r`` are combined localization
  quantities. They are model outputs, not probabilities.
* If detailed analysis is available, names such as ``loc_b_r_s_delta_l`` and
  ``cum_loc_b_r_s`` report localization widths and cumulative localization.
  This extra analysis can be disabled with ``detailed_analysis_flag=False``.

Use xarray's named dimensions to select data. For example,
``analysis.electron_density.sel(tau=...)`` selects the sample nearest a chosen
``tau`` coordinate; ``.isel(tau=...)`` selects by integer index. For matrix
components, select labels such as ``row="R"`` and ``col="Z"`` instead of
relying on their storage order.

Output files
------------

By default, Scotty writes ``scotty_output.h5`` in the current directory. The
``output_path`` argument chooses another directory, and
``output_filename_suffix`` is appended to the filename. For example,
``output_path="results"`` and ``output_filename_suffix="_example"`` produce
``results/scotty_output_example.h5``. Create the output directory before the
run if it does not already exist.

The file uses the HDF5 container and ``h5netcdf`` engine because Scotty stores
complex beam quantities. To load results from a prior run, install the package
dependencies and use `xarray.open_datatree
<https://docs.xarray.dev/en/stable/generated/xarray.open_datatree.html>`_ with
``engine="h5netcdf"`` as shown above. See :ref:`input` for the input profile
and equilibrium formats.
