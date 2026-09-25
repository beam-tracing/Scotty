.. _getting_started:

Getting started
===============

This page takes you through a first Scotty simulation, from installation to
opening the results. It uses the built-in synthetic diagnostic, so it does not
require an experimental equilibrium or density file.

Install Scotty
--------------

Scotty requires Python 3.10 or newer. Install the published package with
``pip``:

.. code-block:: console

   python -m pip install scotty-beam-tracing

The package is named ``scotty-beam-tracing``, but its Python import name is
``scotty``. To work from a local checkout instead, install it in editable mode
from the repository root:

.. code-block:: console

   python -m pip install -e .

Run a synthetic simulation
--------------------------

The ``DBS_synthetic`` preset describes an analytical, circular-flux-surface
case and supplies its own density profile. It is a useful first run because it
does not depend on machine-specific input files.

Save the following as ``run_synthetic.py`` and run it with Python:

.. code-block:: python

   from pathlib import Path

   from scotty import beam_me_up, get_parameters_for_Scotty

   output_dir = Path("scotty-results")
   output_dir.mkdir(parents=True, exist_ok=True)

   parameters = get_parameters_for_Scotty("DBS_synthetic")
   parameters.update(
       output_path=output_dir,
       output_filename_suffix="_demo",
       figure_flag=False,
   )

   results = beam_me_up(**parameters)
   print(results)

The returned object is an ``xarray.DataTree``. It contains the simulation
inputs, solver results, and derived analysis in separate groups. For example,
the ray trajectory is stored in ``solver_output``:

.. code-block:: python

   trajectory = results.solver_output
   print(trajectory.q_R)
   print(trajectory.q_zeta)
   print(trajectory.q_Z)

The coordinates ``q_R``, ``q_zeta``, and ``q_Z`` are the radial, toroidal, and
vertical components of the trajectory. They are sampled along ``tau``, the
parameter along the beam.

Read the saved results
----------------------

By default, Scotty writes a file named ``scotty_output.h5`` in the current
directory. In this example, the output directory and suffix make the filename
``scotty-results/scotty_output_demo.h5``. The HDF5 file stores the same
DataTree, including complex-valued beam quantities:

.. code-block:: python

   import xarray as xr

   saved_results = xr.open_datatree(
       "scotty-results/scotty_output_demo.h5", engine="h5netcdf"
   )
   print(saved_results)

The main groups are:

``inputs``
   Parameters and grids used to run the simulation.
``solver_output``
   The trajectory, wavevector components, and evolving ``Psi_3D`` beam tensor.
``analysis``
   Derived quantities, including beam geometry and localization results.

For a first look, compare the ray coordinates with the density and poloidal
flux along the same samples:

.. code-block:: python

   print(saved_results.solver_output.q_R)
   print(saved_results.analysis.electron_density)
   print(saved_results.analysis.poloidal_flux)

The arrays use labelled coordinates, so you can select by the ``tau``
coordinate or by index. The ``tau`` coordinate is the solver's integration
parameter, not physical distance; use ``analysis.distance_along_line`` for
distance along the ray. See :ref:`output` for how to interpret the main
quantities and :ref:`input` when replacing the synthetic case with files from
an experiment.

Next steps
----------

* See :ref:`input` and the :func:`scotty.beam_me_up.beam_me_up` API
  documentation for the available simulation parameters.
* The :func:`scotty.init_bruv.get_parameters_for_Scotty` API documents
  diagnostic presets.
* Look through ``Examples/`` in the repository for diagnostic-specific runs
  and parameter sweeps. Many of these examples use local experimental data
  paths, which you will need to adapt.
* To run the test suite from a checkout, install the test extra and run
  ``pytest -v``.

.. note::

   ``get_parameters_for_Scotty`` also has presets for real diagnostics. Those
   runs generally need the corresponding magnetic-equilibrium and density data;
   the synthetic preset above is the self-contained starting point.
