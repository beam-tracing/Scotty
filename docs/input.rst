.. _input:

Inputs and data files
=====================

Scotty needs beam launch settings, a magnetic equilibrium, and an electron
density profile. Some runs also use an electron-temperature profile. Start with
the self-contained :ref:`getting_started` example before connecting
machine-specific data.

Set up an experimental run
--------------------------

The main entry point is :func:`scotty.beam_me_up.beam_me_up`. A diagnostic
preset from :func:`scotty.init_bruv.get_parameters_for_Scotty` supplies
diagnostic-specific defaults, such as nominal beam width and launch position.
It does not retrieve experimental data or infer all shot-specific launch
settings. For each run, set and verify the launch frequency, poloidal and
toroidal launch angles, and mode (``+1`` for O-mode, ``-1`` for X-mode).

For example, this starts a DIII-D run using an OMFIT-exported equilibrium and
a separate density profile. Replace the illustrative launch settings, shot
suffix, and paths with the values for your case:

.. code-block:: python

   from pathlib import Path

   from scotty import beam_me_up, get_parameters_for_Scotty

   parameters = get_parameters_for_Scotty(
       "DBS_UCLA_DIII-D_240",
       launch_freq_GHz=55.0,
       find_B_method="omfit",
   )
   parameters.update(
       poloidal_launch_angle_Torbeam=5.0,
       toroidal_launch_angle_Torbeam=-3.0,
       mode_flag=-1,
       density_fit_method="smoothing-spline-file",
       magnetic_data_path=Path("inputs/equilibrium"),
       ne_data_path=Path("inputs/profiles"),
       input_filename_suffix="_shot1",
       output_path=Path("results"),
   )

   Path("results").mkdir(parents=True, exist_ok=True)
   results = beam_me_up(**parameters)

In Scotty's TORBEAM angle convention, a positive poloidal launch angle points
downwards. Check the convention when translating angles from another diagnostic
or code. The example values above are illustrative, not validated settings for
a particular shot.

With ``input_filename_suffix="_shot1"``, the example expects:

.. code-block:: text

   inputs/
   ├── equilibrium/
   │   └── topfile_shot1.json
   └── profiles/
       └── ne_shot1.dat

The suffix is a filename convention: Scotty appends it to the input file
names. It does not select the shot or time inside the data. Choose an
equilibrium snapshot and density profile from the same shot/time, and make
sure the launch settings and flux-boundary parameters agree with those data.

OMFIT JSON equilibrium
----------------------

For ``find_B_method="omfit"``, Scotty reads
``topfile{input_filename_suffix}.json`` from ``magnetic_data_path``. This is a
single-time equilibrium snapshot: select the desired time when preparing the
OMFIT export. The JSON supplies the R and Z grids, magnetic-field components,
and poloidal-flux grid. Scotty expects the keys ``R``, ``Z``, ``Br``, ``Bt``,
``Bz``, and ``pol_flux``; the field and flux arrays are flattened grid data, so
using the OMFIT export is preferable to assembling the JSON by hand. The
flattening follows TORBEAM's column-major (Fortran) grid order.

The OMFIT file provides the magnetic equilibrium, not the density profile.
Scotty still reads a separate ``ne{input_filename_suffix}.dat`` from
``ne_data_path`` when using ``density_fit_method="smoothing-spline-file"``.
The density file's first line is ignored (commonly it contains the number of
profile points). The following rows have two whitespace-separated columns:

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

These values illustrate the file format, not a recommended profile. Use enough
points to represent your measured profile smoothly. The default interpolation
order is 5 and requires at least six points. ``poloidal_flux_zero_density``
sets the flux label at and beyond which Scotty sets the density to zero; ensure
it is consistent with the equilibrium and profile.

TORBEAM-format files
--------------------

If you already use TORBEAM, set ``find_B_method="torbeam"``. Scotty reads
``topfile{input_filename_suffix}`` from ``magnetic_data_path``; it contains the
R and Z grids, magnetic-field components, and poloidal flux. The density file
is read in the same way as described above. Scotty's current equilibrium
loader does not read TORBEAM's ``inbeam.dat`` file.

Optional temperature profile
----------------------------

If ``relativistic_flag=True``, also provide a temperature profile. By default,
Scotty looks for ``Te{input_filename_suffix}.dat`` in ``Te_data_path``. Its
first column is :math:`\sqrt{\psi}` and its second column is temperature in
keV. Relativistic corrections are not needed for the synthetic first run.

Paths, suffixes, and other formats
----------------------------------

* ``magnetic_data_path``, ``ne_data_path``, and ``Te_data_path`` are separate
  directories. Set them explicitly rather than relying on machine-specific
  paths in old example scripts.
* ``input_filename_suffix`` is appended to input names. For example,
  ``"_shot1"`` gives ``topfile_shot1.json`` with the OMFIT reader and
  ``ne_shot1.dat`` for a file-based density profile. It is independent of
  ``output_filename_suffix``.
* Relative paths are resolved from the directory where you run Python, not
  from the directory containing your script. Use ``pathlib.Path`` or absolute
  paths to make this clear.
* Other magnetic-geometry readers include EFIT++ and saved UDA data. These
  formats have different file requirements and may require a shot number or
  equilibrium time. Check the ``find_B_method`` documentation in
  :func:`scotty.beam_me_up.beam_me_up` and the relevant example before using
  them.
* The synthetic ``analytical`` method requires no equilibrium file; its
  geometry and density profile are included in the ``DBS_synthetic`` preset.

For the complete list of beam and solver parameters, see the
:func:`scotty.beam_me_up.beam_me_up` API documentation.
