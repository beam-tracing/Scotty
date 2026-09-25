.. _output:

Plotting and reading results
============================

Scotty returns an ``xarray.DataTree`` and saves it to an HDF5 file. A good
first look at a run is the plotting workflow used in ``Examples/plot.py``: open
the saved tree and pass it to functions from ``scotty.plotting``.

Open a saved run
----------------

Load the file with the ``h5netcdf`` engine:

.. code-block:: python

   from pathlib import Path

   import matplotlib.pyplot as plt
   import xarray as xr

   from scotty.plotting import (
       plot_dispersion_relation,
       plot_poloidal_beam_path,
       plot_toroidal_beam_path,
   )

   results_file = Path("results/scotty_output_example.h5")
   tree = xr.open_datatree(results_file, engine="h5netcdf")

The result tree has three main groups:

``inputs``
   Run settings, input paths, and the equilibrium grid used in the simulation.
``solver_output``
   The propagated trajectory, wavevector, and complex ``Psi_3D`` beam tensor.
``analysis``
   Derived quantities sampled along the ray, including density, poloidal flux,
   beam geometry, and localization terms.

Plot the ray in the poloidal plane
----------------------------------

The most useful first plot is usually ``plot_poloidal_beam_path``:

.. code-block:: python

   ax = plot_poloidal_beam_path(tree, zoom=True)
   plt.show()

This plot shows poloidal-flux contours from the run's input equilibrium, the
launch position, the central ray through vacuum and plasma, and dashed lines
indicating the beam width. ``zoom=True`` frames the view around the beam. The
function returns a Matplotlib ``Axes``, so you can add annotations or change
labels. It does not mark the cutoff automatically; to annotate Scotty's cutoff
estimate, add it to the plot explicitly:

.. code-block:: python

   analysis = tree["analysis"]
   cutoff_index = int(analysis.cutoff_index.item())
   cutoff_R = analysis.q_R.isel(tau=cutoff_index).item()
   cutoff_Z = analysis.q_Z.isel(tau=cutoff_index).item()

   ax = plot_poloidal_beam_path(tree, zoom=True)
   ax.scatter(cutoff_R, cutoff_Z, marker="x", color="red", label="cutoff estimate")
   ax.legend()
   plt.show()

Scotty's cutoff estimate is the point where the wavevector magnitude is
smallest along the computed trajectory. Treat it as an estimate from the run:
first check that the ray path and equilibrium cover the region of interest.

Other useful plots
------------------

``plot_toroidal_beam_path(tree)``
   Shows the ray and beam width in the Cartesian X-Y (toroidal) plane, along
   with the last closed flux surface and magnetic axis.
``plot_dispersion_relation(tree["analysis"])``
   Compares the absolute dispersion-relation solutions calculated by two
   methods as a function of distance from the cutoff. Use it as a numerical
   check: the physical branch should remain close to zero along the ray. The
   plotted branch labels are not a substitute for checking the selected mode.
``plot_widths(tree)``
   Plots the two principal beam widths against physical distance along the ray.
``plot_instrumentation_functions(tree)``
   Plots the localization factors and their combined response. These are
   model-dependent analysis quantities, not probabilities.
``plot_psi(tree)``
   Plots poloidal flux and the real and imaginary parts of the transverse beam
   tensor components along the solver parameter ``tau``.

The plotting functions are in ``scotty.plotting``. Most accept the whole
DataTree; ``plot_dispersion_relation`` takes the ``analysis`` group. They
generally return Matplotlib axes (or axes arrays), which can be used to
customize the figure. ``plot_psi`` creates its own figure. For batch processing,
pass a filename stem; the functions append ``.png``. Create the output
directory first:

.. code-block:: python

   figure_dir = Path("plots")
   figure_dir.mkdir(parents=True, exist_ok=True)

   plot_poloidal_beam_path(tree, filename=figure_dir / "poloidal", zoom=True)
   plot_toroidal_beam_path(tree, filename=figure_dir / "toroidal")
   plot_dispersion_relation(tree["analysis"], filename=figure_dir / "dispersion")

To create the standard quick-look figures during a simulation, leave
``figure_flag=True`` (the default). Scotty creates poloidal ray and dispersion
figures in ``output_path``, using ``output_filename_suffix`` in their names.
Set ``figure_flag=False`` to skip these figures and make plots later from the
saved DataTree.

Inspect the underlying data
---------------------------

The plotting helpers are a starting point; the underlying values remain
available for custom analysis. For example:

.. code-block:: python

   ray = tree["solver_output"]
   analysis = tree["analysis"]

   print(ray.q_R)                    # major radius, in metres
   print(ray.q_zeta)                 # toroidal angle
   print(ray.q_Z)                    # vertical position, in metres
   print(analysis.electron_density)  # density sampled along the ray
   print(analysis.poloidal_flux)     # flux label sampled along the ray

The trajectory coordinates are cylindrical: ``q_R`` is major radius,
``q_zeta`` is toroidal angle, and ``q_Z`` is height. ``tau`` is the solver's
integration parameter, not physical distance. Use
``analysis.distance_along_line`` for distance from launch and
``analysis.l_lc`` for signed distance from the estimated cutoff. The
``analysis.beam_cartesian`` variable contains Cartesian trajectory coordinates.

Use xarray's named dimensions to select values: ``.isel(tau=index)`` selects a
sample by integer index, while ``.sel(tau=value, method="nearest")`` selects
the sample nearest a coordinate value. For matrix-valued data such as
``Psi_3D``, use its ``row`` and ``col`` labels (``"R"``, ``"zeta"``, ``"Z"``)
rather than relying on array order. ``Psi_3D`` is complex-valued; inspect its
real and imaginary components separately when needed.

Output files
------------

By default, Scotty writes ``scotty_output.h5`` in the current directory. The
``output_path`` argument selects another directory, and
``output_filename_suffix`` is appended to the file name. For example,
``output_path="results"`` and ``output_filename_suffix="_example"`` produce
``results/scotty_output_example.h5``. Create the directory before the run if
it does not already exist.

The HDF5 file uses ``h5netcdf`` because Scotty stores complex beam quantities.
Reopen it with `xarray.open_datatree
<https://docs.xarray.dev/en/stable/generated/xarray.open_datatree.html>`_ and
``engine="h5netcdf"`` as shown above. Close the tree when finished with the
file:

.. code-block:: python

   tree.close()

See :ref:`input` for details on the equilibrium and profile files used to
produce a result.
