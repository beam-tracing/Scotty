.. _code-layout:

Code layout
===========

Scotty's code follows the path of the beam: from the antenna, through the
plasma, to what a diagnostic measures. Each stage has its own place in the
code and its own group in the output, so a new feature has an obvious home
and doesn't need to edit the others.

.. list-table::
   :header-rows: 1
   :widths: 25 35 20

   * - Stage
     - Code
     - Output group
   * - The antenna
     - `scotty.antenna`
     - ``inputs``
   * - Entering the plasma
     - `scotty.launch`
     - ``solver``
   * - Tracing the ray and beam
     - `scotty.ray_solver`, `scotty.beam_solver`
     - ``solver``
   * - The beam along the ray
     - `scotty.analysis`
     - ``beam``
   * - What a diagnostic measures
     - `scotty.diagnostics`
     - ``diagnostics/<name>``

The antenna
-----------

`scotty.antenna` holds everything about the beam before it reaches the
plasma: the beam the antenna launches (its :math:`\Psi_w`), the horn, the
lenses and mirrors, and the beam's propagation through vacuum between them.

- `scotty.antenna.gaussian`: the launched circular Gaussian beam's
  :math:`\Psi_w`, and Gaussian beams in vacuum (waist, Rayleigh length,
  propagation)
- `scotty.antenna.horns`: horns, and the Gaussian beam each one launches
- `scotty.antenna.lenses`: lenses and other ABCD optics, acting on
  :math:`\Psi_w`

Antenna code doesn't know about the plasma. What it works out reaches
`beam_me_up` as the launch parameters (``launch_beam_width`` and
``launch_beam_curvature``), which are stored in ``inputs``. The same models
describe a receiving antenna, since an antenna receives the beam it would
launch. The settings of particular machines' antennas, such as
``launch_beam_DBS_UCLA_MAST_U_thin_lens``, stay with the other presets in
`scotty.init_bruv` and use the functions here.

New kinds of launched beam (astigmatic, higher-order modes, heating and
current-drive beams) and other antenna components (polarisers, mirrors) go
here.

Through the plasma
------------------

`scotty.launch` finds where the beam enters the plasma, applies the
boundary conditions there and chooses the mode. The solvers then trace the
ray and the beam through the plasma, and their results are stored in
``solver``. `scotty.analysis` works out the quantities of the beam itself
along the ray, stored in ``beam``: the beam frame, :math:`\theta`, the
widths and the consistency checks such as ``H_Booker``.
These are needed whatever is being measured.

Diagnostics
-----------

`scotty.diagnostics` holds what a particular measurement sees, worked out
from the traced beam. There is one module per diagnostic, and each makes one
output group, ``diagnostics/<name>``.

- `scotty.diagnostics.dbs`: Doppler backscattering, stored in
  ``diagnostics/dbs``: the filter functions (``filterfun_*``), the
  backscattered wavenumber and the quantities only DBS needs

Other diagnostics (cross-polarisation scattering, electron cyclotron
emission, reflectometry) go here, each in its own module.

Which one?
----------

- A quantity that describes the hardware, or the beam before the plasma,
  belongs to the antenna.
- A quantity of the beam in the plasma that any diagnostic might use belongs
  to ``beam``, even if only one diagnostic uses it so far. For example,
  :math:`\theta` and ``k_perp1_hat`` are fixed by the beam frame and the
  field, not by DBS.
- A quantity that only makes sense for one measurement, such as a filter
  function, belongs to that diagnostic.

Old import paths
----------------

The code was moved, not changed, so the old import paths still work and give
the same objects: ``scotty.hornpy`` is `scotty.antenna.horns`,
``scotty.lensalot`` is `scotty.antenna.lenses`, the Gaussian-beam functions
of ``scotty.fun_general`` (``find_waist``, ``find_Rayleigh_length``,
``find_distance_from_waist``, ``propagate_beam`` and
``propagate_circular_beam``) are those of `scotty.antenna.gaussian`, and
``scotty.analysis.dbs_analysis`` is `scotty.diagnostics.dbs.dbs_analysis`.
