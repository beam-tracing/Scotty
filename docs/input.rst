.. _input:

Inputs and data files
=====================

Scotty needs beam launch settings, a magnetic equilibrium, and an electron
density profile. Some runs also use an electron-temperature profile. The
:ref:`getting_started` example uses an analytical equilibrium and density
profile, so it is a good way to check an installation before setting up
machine-specific data.

Choose and review parameters
----------------------------

The main entry point is :func:`scotty.beam_me_up.beam_me_up`. It accepts the
full set of launch, equilibrium, profile, solver, and output settings as
keyword arguments. Use
:func:`scotty.init_bruv.get_parameters_for_Scotty` to start from a diagnostic
preset, then set any values specific to your shot and data. Presets do not
download experimental data, and some leave values such as launch angles or
polarisation unset. Review the returned dictionary and fill in any ``None``
values before running a real case.

For example, the following prepares TORBEAM-format inputs. Replace the example
launch values and file paths with values for your diagnostic:

.. code-block:: python

   from pathlib import Path

   from scotty import beam_me_up, get_parameters_for_Scotty

   parameters = get_parameters_for_Scotty(
       "DBS_NSTX_MAST",
       launch_freq_GHz=52.0,
       find_B_method="torbeam",
   )
   parameters.update(
       poloidal_launch_angle_Torbeam=-5.0,
       toroidal_launch_angle_Torbeam=0.0,
       mode_flag=1,  # +1: O-mode; -1: X-mode
       density_fit_method="smoothing-spline-file",
       magnetic_data_path=Path("inputs/equilibrium"),
       ne_data_path=Path("inputs/profiles"),
       input_filename_suffix="_example",
       output_path=Path("results"),
   )

   Path("results").mkdir(parents=True, exist_ok=True)
   results = beam_me_up(**parameters)

With ``input_filename_suffix="_example"`` and the paths above, Scotty looks
for these files:

.. code-block:: text

   inputs/
   ├── equilibrium/
   │   └── topfile_example
   └── profiles/
       └── ne_example.dat

The example is a template, not a complete experimental setup. In particular,
the beam settings and equilibrium/profile data must describe the same shot,
time, and coordinate convention.

TORBEAM-format files
--------------------

With ``find_B_method="torbeam"``, Scotty reads the magnetic equilibrium from
``topfile{input_filename_suffix}`` in ``magnetic_data_path``. The file contains
the :math:`R` and :math:`Z` grids, magnetic-field components, and poloidal flux
on that grid. The TORBEAM ``inbeam.dat`` file is not read by Scotty's current
equilibrium loader.

For the density, the file ``ne{input_filename_suffix}.dat`` is read from
``ne_data_path`` when using the ``smoothing-spline-file`` fit. Its first line
is ignored (commonly it contains the number of profile points). The following
rows have two whitespace-separated columns:

1. :math:`\rho = \sqrt{\psi}`, the square root of the normalised poloidal-flux
   label;
2. electron density in units of :math:`10^{19}\,\mathrm{m}^{-3}`.

For example:

.. code-block:: text

   6
   0.0  4.0
   0.2  3.8
   0.4  3.2
   0.6  2.2
   0.8  1.0
   1.0  0.0

The values above illustrate the format, not a recommended profile. Use enough
points to represent your measured profile smoothly. The default interpolation
order is 5 and requires at least six points. ``poloidal_flux_zero_density``
sets the flux label at and beyond which Scotty sets the density to zero. It
must be consistent with the profile and equilibrium you provide.

If ``relativistic_flag=True``, provide a temperature profile as well. By
default Scotty looks for ``Te{input_filename_suffix}.dat`` under
``Te_data_path``. Its first column is also :math:`\sqrt{\psi}`; the second
contains temperature in keV. Relativistic corrections are not needed for the
synthetic first run.

Paths and suffixes
------------------

* ``magnetic_data_path``, ``ne_data_path``, and ``Te_data_path`` are separate
  directories. Set them explicitly rather than relying on machine-specific
  paths from an example script.
* ``input_filename_suffix`` is appended to the equilibrium/profile file names.
  Use the same suffix for the files belonging to one input set. It is
  independent of ``output_filename_suffix``.
* Relative paths are resolved from the directory where you run Python, not
  from the directory containing your script. Use ``pathlib.Path`` or absolute
  paths to make this clear.

Other equilibrium formats
-------------------------

The code also supports other magnetic-geometry readers, including OMFIT JSON,
EFIT++ and saved UDA data. These methods have different file names and may
require a shot number or equilibrium time. Check the ``find_B_method``
documentation in :func:`scotty.beam_me_up.beam_me_up` and the relevant example
before selecting one. The synthetic ``analytical`` method needs no equilibrium
file; its geometry parameters are included in the ``DBS_synthetic`` preset.

For the complete list of beam and solver parameters, see the
:func:`scotty.beam_me_up.beam_me_up` API documentation.
