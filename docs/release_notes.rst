Release notes
=============

Unreleased
----------

Packaging
~~~~~~~~~

- A source archive without ``.git`` (such as GitHub's source zip of a
  release) installs: ``.git_archival.txt`` gives ``setuptools_scm`` the
  version. Installing 4.0.0rc1 from its source zip failed.
- Builds no longer warn: the version comes from ``setuptools_scm`` alone, and
  the subpackages ``scotty.antenna`` and ``scotty.diagnostics`` are listed.
  The installed files are the same.
- Tested on Python 3.13 too. ``test_beam_me_up_docstring_covers_every_argument``
  failed there, because Python 3.13 strips the indentation of docstrings.

Documentation
~~~~~~~~~~~~~

- The docstrings and comments were checked against the code, and those that
  were wrong were fixed: for example, `scotty.ray_solver.ray_tracing` returns
  5 values, not 2, and `scotty.fun_general.find_inverse_2D` warns but does not
  raise. The public functions and classes that had no docstring have one, and
  every docstring uses the numpydoc style (``Parameters`` / ``----------`` /
  ``name : type``).
- The README says what Scotty does, its status, how to install and run it,
  how to move from v3, how to contribute and get help, how to cite it, and
  its license.
- The API pages list every module (9 were missing), and the input page shows
  the docstring of `scotty.beam_me_up.beam_me_up`. The documentation takes its
  version from the installed package, links to ``beam-tracing/Scotty``, and
  makes a name between single backquotes a link to the Python object of that
  name, if there is one, with no warning if there is none.

v4.0.0rc1
---------

The first release candidate of Scotty 4.0.0. It freezes the interface: the
argument names and their defaults, what the inputs mean, and the output
variables and what they mean. Between the rc and 4.0.0, only fixes that leave
these alone go in. Please report problems at
https://github.com/beam-tracing/Scotty/issues.

Fixes from the code audit of November 2026. Several change results, as noted.

Results that change
~~~~~~~~~~~~~~~~~~~

- ``QuadraticFit`` is :math:`n_{e0}[1 - (\psi/\psi_0)^2]`, zero at
  :math:`\psi_0`. It was :math:`n_{e0}(1 - \psi^2/\psi_0)` (as in v3), zero at
  :math:`\sqrt{\psi_0}` and negative just inside :math:`\psi_0`, which made runs
  fail. Results change for :math:`\psi_0 \ne 1`.
- ``StefanikovaFit`` uses :math:`\exp[-(\psi/a_\mathrm{width})^{a_\mathrm{exp}}]`,
  as its docstring and Stefanikova et al. (2016) have. The code computed
  :math:`\exp(-a_\mathrm{exp}\psi/a_\mathrm{width})` (as in v3).
- ``SmoothingSplineFit`` holds its end values past its data instead of
  extrapolating (which could give negative densities), and its ``smoothing``
  defaults to 0, so that it goes through the data. The default smoothing of
  ``beam_me_up`` was already 0.
- Every profile is clipped at zero, with a warning the first time.
- The launch angles are measured from the local :math:`R` direction at the
  antenna, so an antenna at :math:`\zeta \ne 0` (or :math:`Y \ne 0`) gives the
  same beam as one at :math:`\zeta = 0`, turned. They were measured from the
  global :math:`X` axis. Results for an antenna on the :math:`+X` side of the
  :math:`X`-:math:`Z` plane (:math:`\zeta = 0`, as in every cylindrical run
  before) are unchanged; a Cartesian antenna at :math:`X < 0`, :math:`Y = 0` has
  :math:`\zeta = \pi`, so its beam now turns round. See
  `scotty.fun_general.launch_direction_cartesian`, now the only place the
  launch direction is computed.
- ``beam/curvature_1`` and ``curvature_2`` were too small by a factor
  :math:`|g|^2` (about 3e-6). The widths were right.
- ``from_v3_kwargs``: a v3 script that leaves ``vacuum_propagation_flag`` unset
  now gets ``launch_flag="vacuum_analytic"`` (v3's default was numerical
  propagation, which is not converged across the kink in the density at the
  plasma edge), and ``Psi_BC_flag="continuous"`` if it leaves that unset. Its
  ``poloidal_flux_enter``, if unset, stays unset, so the plasma starts where
  the density starts (below), going in from v3's ``poloidal_flux_zero_density``
  (1.01 by default), except for a density that is a plain function and for a
  plasma launch, which keep v3's 1.0. A v3 script whose density does not reach
  zero at its edge (for example an ``ne.dat`` file that ends with a non-zero
  density) is then refused by the continuous condition, and the error says
  what to use. Before, such a script failed on the negative density the spline
  extrapolated to.
- :math:`\epsilon_{12}` (``epsilon_g``) has the opposite sign:
  :math:`\epsilon_{12} = -\Omega_{pe}^2\Omega_{ce}/[\Omega(\Omega^2 -
  \Omega_{ce}^2)]`, with :math:`\Omega_{ce} = eB/m_e > 0` (the electron charge
  stays unsigned), negative for :math:`\Omega > \Omega_{ce}`. With fields
  :math:`\propto \exp(-i\Omega t)`, the electrons, of charge :math:`-e`, give
  the dielectric tensor a term :math:`-i\{\Omega_{pe}^2\Omega_{ce}/[\Omega
  (\Omega^2 - \Omega_{ce}^2)]\}\hat{\mathbf{b}}\times\mathbf{1}`, which Scotty
  writes as :math:`+i\epsilon_{12}\hat{\mathbf{b}}\times\mathbf{1}`. Before, the
  polarisations were the complex conjugates of the right ones: along
  :math:`\mathbf{B}`, :math:`\hat{\mathbf{u}}_1 + i\hat{\mathbf{u}}_2`, which
  rotates with the electrons, was given to the wave with
  :math:`N^2 = 1 - \Omega_{pe}^2/[\Omega(\Omega + \Omega_{ce})]`, instead of
  the one with :math:`N^2 = 1 - \Omega_{pe}^2/[\Omega(\Omega - \Omega_{ce})]`
  that resonates at :math:`\Omega \to \Omega_{ce}`. Hall-Chen, Parra & Hillesheim (2022),
  eqs. (2) and (35), have the same sign error. The
  dispersion relation depends only on :math:`\epsilon_{12}^2`, so the rays, the
  beams, the widths, the curvatures and the DBS weighting (``filterfun_p``,
  which depends only on :math:`|\hat{\mathbf{e}}^* \cdot (\boldsymbol{\epsilon}
  - \mathbf{1}) \cdot \hat{\mathbf{e}}|`) are unchanged. ``epsilon_g`` changes
  sign, and ``e_hat``, ``e_eigvecs`` and ``solver/e_hat_initial`` become their
  complex conjugates (up to an overall sign, which was arbitrary before too).
- ``get_parameters_for_Scotty("DBS_synthetic", launch_freq_GHz)`` uses the
  frequency given; it returned 55 GHz whatever was given. Presets that need a
  frequency raise a clear error without one.

Where the plasma starts
~~~~~~~~~~~~~~~~~~~~~~~

- ``poloidal_flux_enter`` defaults to ``None``: where the density starts,
  going in from its zero point (it was 1.0), so that the boundary condition is
  applied there and the beam never crosses the start of the density inside
  the plasma, where no boundary condition is applied. For a profile from data
  (`scotty.profile_fit.SmoothingSplineFit`) that end inside its zero point, it
  goes in from the end of the data, ``poloidal_flux_data_end``: beyond it the
  profile holds its last value, with a kink there. The output's ``inputs``
  record the entry point used.
- The boundary condition to use depends on how the density starts:

  =====================================  ===================  ====================  ========
  Density where it starts                ``"discontinuous"``  ``"continuous"``      ``None``
  =====================================  ===================  ====================  ========
  jumps (finite)                         yes                  no                    no
  kink (zero, with a finite gradient)    yes                  yes (exact)           no
  zero, with zero gradient               yes                  yes (as ``None``)     yes
  =====================================  ===================  ====================  ========

  The simplest choice, which needs no knowledge of the edge, is
  ``"discontinuous"`` with ``poloidal_flux_enter`` slightly inside where the
  density starts (0.01 inside, say). The thin layer left outside is treated as
  vacuum, which changes little where the edge density is small: in a test
  with a kink, entering from 0.001 to 0.1 inside changed the beam width at the
  cut-off by less than 0.01%.
- ``poloidal_flux_zero_density`` and ``poloidal_flux_zero_temperature`` are
  deprecated, and will be removed after 4.x: give the zero point to the
  profile object instead (the first argument of the fits in
  `scotty.profile_fit`). Giving either warns (``FutureWarning``). Their
  meaning is unchanged, as is that of a named fit's parameters. They default
  to ``None``: the zero point of the profile object, if it has one, and 1.0
  otherwise. A value that disagrees with the profile object's raises; before,
  it was ignored, and the output recorded it anyway.

Errors where results were wrong without one
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``Psi_BC_flag="continuous"`` and ``None`` raise if the density is not zero at
  the plasma entry point (more than 1e-5 of its maximum). Both keep the vacuum
  :math:`K` there, so the ray started off the dispersion relation (:math:`|H|`
  up to 0.08) and stayed there, with at most a warning. The message says what
  to use instead (the table above).
- Every boundary condition raises if the density is zero over a band inside
  the entry point and starts with a kink or a jump: the beam would cross it in
  the plasma with no boundary condition, and an adaptive solver can step over
  it. In one case the width at the cut-off was 0.107 m at the default
  tolerance against 0.0498 m converged, and right only at ``rtol`` 1e-5 or
  below. The message gives the entry point and boundary condition to use. A
  profile that starts with zero gradient is accepted.
- Every boundary condition raises if the density is zero (or negative, which
  the profile clips to zero) anywhere inside where it starts, as a spline
  through data padded with zeros can give.
- Every boundary condition raises if the entry point is beyond the end of a
  spline's data, and the profile is not flat there: beyond its data the
  profile holds its last value, with a kink at the end of the data. In one
  case entering beyond it gave a width at the cut-off of 0.113 m against
  0.0454 m.
- In `scotty.boundary_conditions`, ``check_density_at_entry`` replaces
  ``check_density_zero_at_entry``, and ``check_density_smooth_at_edge`` takes
  the flux to check as ``edge`` (it was ``poloidal_flux_zero_density``).
  `scotty.launch.find_plasma_entry_parameters` no longer takes
  ``poloidal_flux_zero_density``.
- ``Psi_BC_flag=None`` checks that the density is flat where the beam enters,
  not at the profile's zero point.
- A NaN or infinite ``launch_beam_width`` or ``launch_beam_curvature`` raises.

Inputs
~~~~~~

- ``find_B_method`` and ``density_fit_method`` have no defaults, so every
  script names where its field and density come from. Before, an unset
  ``find_B_method`` read a TORBEAM topfile, and an unset ``density_fit_method``
  read ``ne.dat`` or guessed the fit from the number of parameters. To read
  ``ne.dat``, pass ``density_fit_method="smoothing-spline-file"``.
  ``temperature_fit_method`` stays optional, since only relativistic runs use
  it, but ``relativistic_flag=True`` now needs it (it read ``Te.dat`` before).
  ``get_parameters_for_Scotty`` sets both for ``"DBS_synthetic"``, and sets
  ``density_fit_method`` for the machine presets when it is given
  ``find_B_method`` (``"smoothing-spline-file"`` for ``"torbeam"``).
  ``from_v3_kwargs`` sets v3's defaults, with a warning.
- ``magnetic_data_path``, ``ne_data_path`` and ``Te_data_path`` are the
  paths of the files themselves (a full path, or a path relative to the
  current folder), not folders, and ``input_filename_suffix`` is removed. For
  example ``ne_data_path="C:/data/ne_45000.dat"`` with
  ``density_fit_method="smoothing-spline-file"``, and
  ``magnetic_data_path="C:/data/topfile_45000"`` with
  ``find_B_method="torbeam"`` (``"omfit"`` reads the JSON form). The default
  is ``None``, and a path is checked only when the run reads it: a folder
  raises an error that gives an example file name. The output's ``inputs``
  save the paths of the files that the run read (``"None"`` for none).
  For a field read from a file by one of the constructors below,
  ``magnetic_data_path`` is that file, and the new ``magnetic_data_time`` is
  the time the constructor took (the time in the file nearest the one asked
  for; NaN for a file with one time, or no file). They replace v3's inputs
  ``shot`` and ``equil_time``. The constructors set them on the field as
  ``data_file`` and ``data_time``.
- One spline order and smoothing for each of the magnetic field, the density
  and the temperature: ``interp_order_magnetic``,
  ``interp_smoothing_magnetic``, ``interp_order_ne``, ``interp_smoothing_ne``,
  ``interp_order_Te`` and ``interp_smoothing_Te`` (defaults 5 and 0, as v3's)
  replace ``interp_order`` and ``interp_smoothing``, which set all three. An
  invalid order raises; before, it was replaced by 5, with a warning.
- ``"smoothing-spline-file"`` takes no parameters: its file and spline
  settings have their own arguments, and ``density_fit_parameters`` with it
  raises. Before, a file in ``density_fit_parameters`` always raised "expected
  filename as second parameter" (as in v3).
- ``find_B_method`` ``"test"``, ``"test_notime"``, ``"UDA_saved"`` and
  ``"EFITpp"`` are removed, with ``shot`` and ``equil_time``, which only they
  used. Their files are read by field constructors, which take the file and
  the time, and whose field is given to ``find_B_method``:

  - ``"UDA_saved"``: `scotty.geometry.EFITField.from_MAST_saved` (MAST) and
    `scotty.geometry.EFITField.from_MAST_U_saved` (MAST-U; with no
    ``equil_time`` for a file with one time, v3's ``"UDA_saved"`` with no
    shot)
  - ``"EFITpp"``: `scotty.geometry.EFITField.from_EFITpp`
  - ``"test"`` and ``"test_notime"``: the new
    `scotty.geometry.InterpolatedField_Cylindrical.from_npz`, with and
    without ``equil_time``

  Each name raises an error that names its constructor. Before, one name
  chose between file layouts by the shot number (MAST up to 30471, else
  MAST-U), and ``"test_notime"``, and ``"UDA_saved"`` with no shot, took the
  path of a file, which ``beam_me_up`` refused, so they could not run. The
  constructors' spline and step arguments now have defaults (those of
  `scotty.geometry.EFITField`), and a constructor given a file with several
  times and no ``equil_time``, or the reverse, raises.
- `scotty.geometry.EFITField.from_MAST_U_saved` puts ``rBphi`` on a
  normalised flux evenly spaced from 0 to 1 at every time. It used the file's
  ``poloidalFlux``, which in the MAST-U files saved from UDA goes from 1 to 0
  and back to 1, so it failed on them ("x must be strictly increasing"); a
  file with one time already used the evenly spaced flux.
- The removed arguments (``interp_order``, ``interp_smoothing``,
  ``input_filename_suffix``, ``shot``, ``equil_time``) raise an error that
  names what replaces them, rather than the warning for an unknown keyword:
  ignoring them would change the run. ``from_v3_kwargs`` converts them: it
  builds the paths of v3's files (``topfile{suffix}``, ``ne{suffix}.dat``,
  ``Te{suffix}.dat`` in v3's folders), gives v3's one order and smoothing to
  each spline, moves a file in ``density_fit_parameters`` to ``ne_data_path``,
  and drops ``shot`` and ``equil_time``, with warnings.
- ``make_fit`` is in `scotty.profile_fit` (it was in `scotty.beam_me_up`), and
  has no ``filename`` argument.
- New ``analysis_flag`` and ``dbs_analysis_flag``, both ``True`` by default.
  ``analysis_flag=False`` saves only the inputs and the solver output (no
  ``beam`` group); ``dbs_analysis_flag=False`` leaves out the
  ``diagnostics/dbs`` group and the variables of ``beam`` that the DBS analysis
  computes (``scotty.data_io.BEAM_FROM_DBS``; they include ``beam_waist``,
  ``theta`` and ``normal_hat``). The DBS analysis
  needs the beam analysis. Both flags are stored in the inputs.

Presets
~~~~~~~

- ``get_parameters_for_Scotty`` takes ``data_path``, the folder of the input
  files, instead of ``user``, which chose folders on particular computers.
  With ``find_B_method="torbeam"`` it gives the paths of the files in that
  folder, ``topfile`` and ``ne.dat``. ``shot`` and ``equil_time`` only choose
  a saved density fit (``find_ne_method``), with any ``find_B_method`` except
  ``"torbeam"``, and are not passed on to ``beam_me_up``.
- ``DBS_synthetic`` no longer sets the zero points or the entry point: its
  profile object decides, and the plasma starts where its density does (1.0).

Outputs
~~~~~~~

- ``width_1`` and ``width_2`` each follow their own principal axis of
  :math:`\mathrm{Im}(\Psi_w)` along the beam, and so do ``curvature_1`` and
  ``curvature_2`` for :math:`\mathrm{Re}(\Psi_w)`
  (`scotty.fun_general.follow_principal_axes`). ``width_1`` is the larger width,
  and ``curvature_1`` the smaller curvature, where the two differ most; from
  there the axes are matched from point to point by the overlaps of their
  eigenvectors. Before, the values were sorted at each point, so where the
  widths crossed they exchanged axes, each with a cusp. The function meant to
  follow them, ``check_eigval_order``, did nothing and is removed.

  The widths really cross only where :math:`\Psi_{xy}` vanishes too, by
  symmetry (for example with no poloidal field and no toroidal launch angle),
  and there their axes don't turn. Otherwise they come close and turn away
  while the axes turn by about 90° (an avoided crossing), and each width
  follows its turning axis: then the labels are the sorted ones, as before. So
  the order of the two is exchanged only where, in the axes of the last point
  where the order was decided, the off-diagonal element is below 1e-4 of the
  size: true crossings give 1e-8 to 1e-6, and an avoided crossing at least half
  its closest gap, however coarse the output steps. Widths within 1e-4 of each other (relative) are
  taken as equal, and the axes undefined. Where they are within 1%, as in a
  nearly circular beam, noise can set the axes, and exchanging the widths would
  hardly change them: there the labels follow the axes from point to point,
  and the order isn't decided. Where the axes turn by more than 40° between
  consecutive output points that both differ by more than 1%, a warning says so
  (increase ``len_tau``), and the order is kept there unless the coupling shows
  a true crossing.

  Which width is ``width_1`` is set in each run. In a scan where the widths
  cross, it can be the larger in one run and the smaller in the next.
- The axes are saved as Cartesian unit vectors: ``width_1_direction``,
  ``width_2_direction``, ``curvature_1_direction`` and
  ``curvature_2_direction``, NaN where the two widths (or curvatures) are equal
  and the axes undefined, as for a circular beam. The axes of the widths and of
  the curvatures are in general different.
- `scotty.fun_general.find_beam_widths_and_curvs` works for a single beam
  matrix too (it raised an error before).
- ``solver/e_hat_initial`` is labelled with ``col_field`` = (u1, u2, b), the
  basis it is in, instead of ``col_cart``.
- ``solver/mode_index`` (the traced root of the dispersion tensor where the
  solvers start, in the ascending order of its eigenvalues) is the root whose
  eigenvector is the traced mode's polarisation, ``solver/e_hat_initial``. At
  zero density (an entry point at the plasma edge) two roots vanish together;
  the root with the smaller ``|H|`` was taken, so ``mode_index`` named the
  O-mode's root for the X-mode too, and every such run warned that it could not
  tell the O- and X-modes apart (#185). The warnings are gone. The results of
  the runs did not change.
- Duplicated variables are removed: ``epsilon_bb``, ``epsilon_11`` and
  ``epsilon_12`` (use ``epsilon_para``, ``epsilon_perp`` and ``epsilon_g``);
  ``H_Cardano_1``-``3`` (use ``H_Cardano``); ``H_eigval1``-``3`` (use
  ``H_eigvals``); ``e_eigvec1``-``3`` (use ``e_eigvecs``); ``q_X_output``,
  ``q_Y_output``, ``K_X_output`` and ``K_Y_output`` (use ``q_output_cartesian``
  and ``K_output_cartesian``).

Removed
~~~~~~~

- ``further_analysis_flag`` and ``detailed_analysis_flag``, which did nothing
  in v4; ``from_v3_kwargs`` drops them. ``figure_flag`` stays (it still makes
  no figures, and ``True`` logs a warning): separate flags for separate figures
  may come later. See `scotty.plotting` for figures.
- ``scotty/outplot.py``, which read only v2 output files, and the
  ``TestFunctions``, ``WorkInProgress`` and ``Deprecated`` folders. The ``v3``
  branch keeps them.
- The examples for particular machines, which used data files that are not in
  the repository. ``Examples/`` keeps an analytical case
  (``analytical_circular.py``) and the plotting example (``plot.py``).
- Unused functions in ``fun_general``: ``find_epsilon_para``,
  ``find_epsilon_perp``, ``find_epsilon_g`` (use ``find_epsilon_terms``),
  ``tilt_trns_RZ_make``, ``tilt_trns_TZ_make``,
  ``mirrornorm_make_with_rot_tilt_operators`` and a second copy of
  ``find_mode_index_and_ehat``; and ``mln`` and ``ralt`` in ``logger``.
- ``launch.find_mode_index_and_ehat`` and the ``tol_O_mode_polarisation``
  argument of ``launch.find_plasma_entry_parameters`` (#185). Its branch for
  ``"O"`` and ``"X"`` could not be reached, and it chose the O-mode for both.

Packaging
~~~~~~~~~

- Requires SciPy 1.13 and Matplotlib 3.8 or newer (the first built for NumPy 2,
  which Scotty already required). The project URL is ``beam-tracing/Scotty``.
- The publish workflow fetches the full history, so that the package gets its
  version from the tags.

v4.0.0.dev0
-----------

Scotty v4 is a rewrite of v3. It is a development version: its interface and
outputs may still change, and some of v3's features have not been ported. v3
(latest release 3.2.0) is the stable line; use it for published work, or
anything you need to reproduce later.

Output layout
~~~~~~~~~~~~~

- The output's groups are ``inputs``, ``solver`` (was ``solver_output``), ``beam``
  (``analysis``, including the consistency checks such as ``H_Booker``, and the
  beam quantities of ``dbs_analysis``, such as ``theta`` and ``k_perp1_hat``) and
  ``diagnostics/dbs`` (the rest of ``dbs_analysis``). Variables keep their names,
  apart from the filter functions (below).
- The code follows the same layout (:ref:`code-layout`). `scotty.antenna` holds
  the beam before the plasma: `scotty.antenna.gaussian` (the launched Gaussian
  beam, and the Gaussian-beam functions that were in ``fun_general``),
  `scotty.antenna.horns` (was ``hornpy``) and `scotty.antenna.lenses` (was
  ``lensalot``). `scotty.diagnostics` holds one module per diagnostic:
  `scotty.diagnostics.dbs` has ``dbs_analysis``, which was in ``analysis``. The
  old import paths still work, apart from ``scotty.analysis.dbs_analysis``.

Geometry, launch and boundary conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``geometry`` (required): ``"cylindrical"`` (:math:`R, \zeta, Z`, axisymmetric
  fields, as in v3) or ``"cartesian"`` (:math:`X, Y, Z`, general 3D fields).
  Outputs are given in both coordinate systems.
- ``launch_flag`` (required) replaces v3's ``vacuumLaunch_flag`` and
  ``vacuum_propagation_flag``:

  - ``"vacuum_analytic"``: the beam is launched at the antenna and propagated
    analytically through the vacuum to the plasma boundary, where
    ``Psi_BC_flag`` is applied; the solvers start there. v3's
    ``vacuumLaunch_flag=True``, ``vacuum_propagation_flag=True``.
  - ``"vacuum_numerical"``: the solvers start at the antenna and integrate the
    beam through the vacuum and the plasma boundary, with no boundary
    conditions. The plasma-entry quantities are saved as NaN. v3's
    ``vacuumLaunch_flag=True``, ``vacuum_propagation_flag=False``.
  - ``"plasma"``: the solvers start inside the plasma, at
    ``launch_position``, with ``plasmaLaunch_K_cartesian`` and
    ``plasmaLaunch_Psi_3D_lab_cartesian`` (Cartesian components). v3's
    ``vacuumLaunch_flag=False``, which took K in cylindrical components. The
    launch point must be inside the plasma, and K must satisfy the dispersion
    relation.

- ``Psi_BC_flag`` (required; the same name as in v3) is ``"continuous"``,
  ``"discontinuous"`` or ``None``. v3's deprecated ``True`` and ``False`` now
  raise an error (they were ``"continuous"`` and ``None``). ``None`` now checks
  that the density is zero and flat at the plasma edge, and raises otherwise.
  It must be ``None`` for ``launch_flag="vacuum_numerical"``.
- ``Psi_BC_flag="continuous"`` and ``"discontinuous"`` work whatever the
  direction of :math:`\nabla \psi` at the plasma entry point (#176). The
  tangents of the flux surface used to match the beam across the boundary were
  built in coordinate planes, and became parallel when :math:`\nabla \psi` had no
  Z or no X component, or had :math:`\psi_X = -\psi_Y`: for example a Cartesian
  slab, an entry at the midplane of an up-down symmetric equilibrium, or one
  where :math:`\nabla \psi` is vertical. ``find_Psi_3D_plasma`` then failed with
  a singular matrix, and with ``"discontinuous"`` ``find_K_plasma`` gave a NaN
  normal when :math:`\psi_X = 0`. Both now build the tangents from
  :math:`\nabla \psi`. Elsewhere the results are unchanged, up to rounding.
- ``mode_flag`` can also be ``"O"`` or ``"X"``, decided from the two roots'
  polarisations where the solvers start. Which of 1 and -1 is the O-mode
  depends on the plasma and K; the cut-off check at the plasma boundary
  (``Psi_BC_flag="discontinuous"``) now uses the same test.
- ``launch_beam_width`` and ``launch_beam_curvature`` take a number, for a
  circular beam, as before. A 2-tuple and a 3-tuple are reserved for
  elliptical and astigmatic beams and raise ``NotImplementedError`` for now,
  until the launch beam frame and the meaning of their elements are decided.
  Anything else raises ``ValueError``.

Running v3 inputs and outputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``scotty.compat.from_v3_kwargs`` converts the arguments of a v3 script:
  ``beam_me_up(**from_v3_kwargs(kwargs))``. It maps ``vacuumLaunch_flag`` and
  ``vacuum_propagation_flag`` to ``launch_flag`` (with v3's defaults, ``True``
  and ``False``, when they are not given, so a script that does not set
  ``vacuum_propagation_flag`` gets ``"vacuum_numerical"``), passes
  ``Psi_BC_flag`` through, sets ``geometry="cylindrical"``, and converts the
  plasma-launch K to Cartesian components.
- ``scotty.compat.quick_cutoff`` gives what v3's ``quick_run`` returned, from a
  ``ray_tracing_flag=True`` run.
- ``scotty.compat.to_v4_output``, ``to_v3_output`` and ``convert_output_file``
  convert output files between the two layouts, including v3's ``loc_*`` and
  v4's ``filterfun_*``.

Defaults
~~~~~~~~

- The default spatial finite-difference steps ``delta_X``, ``delta_Y``,
  ``delta_R`` and ``delta_Z`` are 1e-5 (v3: 1e-4). The wavevector steps are
  unchanged (0.1).
- The default solver tolerances are ``rtol=1e-4`` and ``atol=1e-7`` (v3: 1e-3,
  1e-6). Use tighter ones (e.g. 1e-6, 1e-9) for anything read close to the
  cut-off, and when :math:`K` at the cut-off is below about a tenth of its
  vacuum value: at about a twentieth, the integrated filter function at the
  default settings was 1% and 22% off on the two launches tried.
- The default number of output points ``len_tau`` is 4002 (v3: 102). The filter
  function is integrated on these points, and where the beam focuses sharply at
  the cut-off its peak can be narrower than their spacing: on one case tried,
  the integral was 62% off at 102 and up to 5% off near 1000. An output file is
  about 10 MB at this ``len_tau``.
- So a run at the default settings can give a noticeably different integrated
  filter function from one at v3's defaults.
- In the beam solver, ``atol`` applies to :math:`q` as given, and to :math:`K`
  and :math:`\Psi` scaled by :math:`|K|` and :math:`\max|\Psi|` at the initial
  point (v3: the same ``atol`` for every component). Components that symmetry
  makes zero no longer force tiny steps to resolve finite-difference noise, which
  made symmetric cases (slabs, cylinders, unsteered midplane entries) slow, and
  could end the beam early. Such cases are now accurate to about ``rtol``; before,
  some were more accurate than that by accident.
- If the beam solver fails (a step can't be taken, e.g. where the beam leaves the
  interpolation region, the density goes negative or the Hamiltonian isn't
  finite), ``beam_me_up`` now raises a ``RuntimeError`` giving how far the beam
  got. Before, it returned the points up to the failure as if the beam had been
  traced to the end, with ``solver_status`` -1 in the output. Runs that passed
  silently with a partial beam, for example at v3's tolerances, may now raise.
- ``from_v3_kwargs`` keeps v3's tolerances, spatial steps and ``len_tau``, so a
  v3 script gives v3's accuracy.
- ``interp_smoothing=0``, ``find_B_method="torbeam"`` and
  ``detailed_analysis_flag=True``, as in v3.
- ``poloidal_flux_zero_temperature`` defaults to ``None``: the temperature
  profile object's zero point, if it has one, and ``poloidal_flux_zero_density``
  otherwise. It must be at least ``poloidal_flux_enter`` only in a
  relativistic run, the only kind that uses the temperature.

Fixes
~~~~~

- The O-mode's Hamiltonian at Booker's :math:`\alpha = \epsilon_\perp = 0`
  (the X-mode's upper hybrid layer): the root was 0/0 there, though the O-mode is
  regular, so an O-mode beam crossing that density with :math:`K` exactly
  perpendicular to :math:`B` could stop the beam solver (#180, #181). ``find_Booker_root`` uses the rationalised form where
  :math:`m\beta > 0`, where it can't cancel. The same root in
  ``boundary_conditions`` and ``launch`` uses it too.

Other interface changes
~~~~~~~~~~~~~~~~~~~~~~~

- The ``DBS_SWIP_MAST-U`` launch beam: its width and curvature were wrong
  (-12.4 m and 24.4 m⁻¹ at 52 GHz, from swapped arguments), and are now
  0.0420 m and -0.331 m⁻¹. They are still estimates, assuming a waist at the
  window and the UCLA DBS's distance to the steering mirror, so give
  ``launch_beam_width`` and ``launch_beam_curvature`` for anything quantitative.

- ``output_path``, ``magnetic_data_path``, ``ne_data_path`` and
  ``Te_data_path`` accept a ``str`` or a ``pathlib.Path``. The output folder is
  created if it does not exist.
- A keyword argument that ``beam_me_up`` does not take is logged as a warning,
  with the closest argument name (``did you mean ...?``).
- The localisation is now called the filter function, in the outputs and the
  code: v3's ``loc_p``, ``loc_r``, ``loc_b``, ``loc_m`` are ``filterfun_p``,
  ``filterfun_r``, ``filterfun_b``, ``filterfun_m``, and their product is
  ``filterfun_all``. v3's ``loc_s`` is not a filter function but the model
  turbulence spectrum, and is now ``delta_ne_sq_<power>``; it is not included in
  ``filterfun_all``. Their path
  integrals are ``int_<name>_dl`` (``l`` is the length along the ray; ``L`` is
  kept for inhomogeneity lengths).
- New arguments (#151, #169): ``first_harmonic_flag``, ``second_harmonic_flag``
  and ``upper_hybrid_flag`` decide what the ray does at each resonance layer: the
  first harmonic of the electron cyclotron resonance (the fundamental), its second
  harmonic, and the upper hybrid resonance. Each is ``"stop"`` (the default: the
  ray stops there and the output is returned), ``"raise"`` (a ``RuntimeError``) or
  ``"pass"`` (the ray carries on through the layer). Each layer has its own solver
  event, so two layers crossed in one solver step are both seen. Temperature
  profiles are built when given, and the layers then include the relativistic
  mass correction.

  - ``first_harmonic_flag`` can't be ``"pass"``: the cold-plasma Hamiltonian is
    singular there, and past it the root that ``mode_flag`` selects swaps (#169).
  - ``upper_hybrid_flag`` applies to the X-mode only (whichever ``mode_flag``
    that is) and is ignored for the O-mode. It can't be ``"pass"``: the X-mode's
    refractive index goes to infinity at the layer, which the ray approaches only
    asymptotically. ``"stop"`` ends the output 1e-2 short of the layer, in
    :math:`\omega_{UH} / \omega`, and ``"raise"`` raises there. Closer, the
    output points are off the dispersion relation (:math:`|H|` up to 1.4 at the
    default ``len_tau``), and with :math:`K` perpendicular to :math:`B` the solver
    stalled and the run didn't return.
  - A ray that starts on the first harmonic raises whatever the flag, and so does
    one that starts on the second harmonic unless its flag is ``"pass"``, or an
    X-mode ray that starts within 1e-2 of the upper hybrid layer.

  They replace ``resonance_flag``, which is removed: passing it raises a
  ``ValueError`` rather than being ignored. ``from_v3_kwargs`` converts v3's
  ``resonance_flag``: ``True`` sets all three flags to ``"stop"``, and ``False``
  sets ``first_harmonic_flag="raise"``, ``second_harmonic_flag="pass"`` and
  ``upper_hybrid_flag="raise"``. v3's ``False`` let the ray carry on past the first
  harmonic and the upper hybrid layer; v4 raises at both.
- Where the ray stops at the first harmonic, the output now ends where
  :math:`\omega_{ce} / \omega` is 1e-3 from 1, about 2 mm before the layer on the
  cases tried. Closer, :math:`K` could jump off the dispersion relation at the
  last few output points (:math:`|H|` up to 0.48, :math:`|K|` 23% off). A ray
  launched within 1e-3 of the layer still ends with such points, and raises from
  ``check_H_output`` with a message about :math:`H`, not the resonance.
- ``check_H_output`` raises above :math:`|H| = 0.1` (was 1) and warns above 0.01
  (#169), so a run that used to finish with a warning may now raise. This
  includes rays that approach the first harmonic from above and stall just short
  of it without crossing it.
- An output with no points from the beam solver raises a ``RuntimeError``, even
  when the solver reports success.
- ``convert_hessians`` takes a required ``K_start`` (Cartesian K).
  ``analysis.py`` is its only caller.
- ``find_B_method="analytical"`` is ``CircularFluxSurfaceField``, as in v3; the
  analytical base class is ``CircularFieldBase``.
- New in ``hamiltonian.py`` and ``fun_general.py``: ``FIRST_ORDER_KEYS``,
  ``find_Booker_discriminant``.
- New outputs: ``filterfun_all`` and ``int_filterfun_all_dl``,
  ``d_xhat_d_tau_dot_yhat``, ``g_hat_Cardano``, ``Psi_w`` on (``row_beam``,
  ``col_beam``) = (x, y), the stacked ``H_Cardano``, and in ``diagnostics/dbs``
  ``k_perp2_bs``, ``k_perp2_bs_normal`` and ``k_perp2_bs_binormal``.
- ``benchmarking_flag`` is removed: it no longer did anything.
- ``e_hat`` and ``e_eigvecs`` are on ``col_field`` = (u1,
  u2, b), the field-aligned frame D is written in, instead of ``col_cart``.
  Their values are unchanged.
- ``solver_output/q_initial_cylindrical`` is where the solvers start: the
  plasma-entry position, or the antenna for ``launch_flag="vacuum_numerical"``.
- ``find_plasma_entry_position``: ``boundary_adjust`` is now the maximum step
  into the plasma.
- ``from scotty import beam_me_up`` and ``get_parameters_for_Scotty`` work, as
  in v3. The presets ``CPS_UCLA_MAST-U`` and ``hiK_Strath_MAST`` raise
  ``NotImplementedError`` (they were never implemented).
- ``scotty.plotting`` has matplotlib versions of v3's
  ``plot_dispersion_relation``, ``plot_poloidal_beam_path`` and
  ``plot_toroidal_beam_path``, for either geometry. v4's output does not store
  the poloidal flux on a grid, so they draw flux surfaces only if the field is
  passed (``beam_me_up(..., return_dt_field=True)`` returns it). The plotly 3D
  plots need plotly, which is not a dependency.

Removed
~~~~~~~

- The filter-function widths (``delta_l``, ``delta_kperp1``, the cumulative
  crossings) and ``find_x0`` from v3 are not in v4; a new version will come in
  a later release, as the old ones were poor and rarely used.
- v3's other matplotlib plots (beam widths, filter functions, 3D beam profile,
  Psi).
- The figures v3 saved with ``figure_flag=True`` (``Ray1_*.png``, ``H_*.png``):
  v4 only stores the flag, which now defaults to ``False`` and is no longer set
  by the presets, and ``True`` logs a warning. Call ``scotty.plotting`` directly.
- The dependency on ``freeqdsk``. Topfiles are now written by ``scotty.torbeam``
  itself, in the same format.
- ``CurvySlabField`` (``find_B_method="curvy_slab"``).
- The analytical fields ``CircularCrossSectionField`` and
  ``ConstantCurrentDensityField``, and ``find_B_method="unit-tests"``; use
  ``"analytical"``.
- ``check_Psi_output``, which warned on every run (it was miscoded); see
  ``check_Psi_g_output`` above.
- ``open_*_npz``.
- The GUI (``GUI/scottypygui.py``) has not been ported to v4.
