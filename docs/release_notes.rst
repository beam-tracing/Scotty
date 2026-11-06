Release notes
=============

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
  old import paths still work.

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
- ``mode_flag`` can also be ``"O"`` or ``"X"``, decided from the two roots'
  polarisations where the solvers start. Which of 1 and -1 is the O-mode
  depends on the plasma and K; the cut-off check at the plasma boundary
  (``Psi_BC_flag="discontinuous"``) now uses the same test.

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
- ``from_v3_kwargs`` keeps v3's tolerances, spatial steps and ``len_tau``, so a
  v3 script gives v3's accuracy.
- ``interp_smoothing=0``, ``find_B_method="torbeam"`` and
  ``detailed_analysis_flag=True``, as in v3.
- ``poloidal_flux_zero_temperature`` defaults to ``None``, which is
  ``poloidal_flux_zero_density``.

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
  ``filterfun_r``, ``filterfun_b``, ``filterfun_m``, and there are
  ``filterfun_all_<power>``. v3's ``loc_s`` is not a filter function but the
  model turbulence spectrum, and is now ``delta_ne_sq_<power>``. Their path
  integrals are ``int_<name>_dl`` (``l`` is the length along the ray; ``L`` is
  kept for inhomogeneity lengths).
- New argument: ``resonance_flag`` (#151). Temperature profiles are built when
  given.
- ``convert_hessians`` takes a required ``K_start`` (Cartesian K).
  ``analysis.py`` is its only caller.
- ``find_B_method="analytical"`` is ``CircularFluxSurfaceField``, as in v3; the
  analytical base class is ``CircularFieldBase``.
- New in ``hamiltonian.py`` and ``fun_general.py``: ``FIRST_ORDER_KEYS``,
  ``find_Booker_discriminant``.
- New outputs: ``int_filterfun_all_<power>_dl``, ``d_xhat_d_tau_dot_yhat``,
  ``g_hat_Cardano``, ``Psi_w`` on (``row_beam``, ``col_beam``) = (x, y), and
  the stacked ``H_Cardano``. ``filterfun_all_<power>`` is the array.
- ``e_hat``, ``e_eigvecs`` and ``e_eigvec1``-``3`` are on ``col_field`` = (u1,
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
- ``check_Psi_output``, which warned on every run.
- ``open_*_npz``.
- The GUI (``GUI/scottypygui.py``) has not been ported to v4.
