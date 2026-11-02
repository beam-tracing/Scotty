Release notes
=============

v4.0.0.dev0
-----------

Scotty v4 is a rewrite of v3. It is a development version: its interface and
outputs may still change, and some of v3's features have not been ported. v3
(latest release 3.2.0) is the stable line; use it for published work, or
anything you need to reproduce later.

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
  v4's ``filter_*``.

Defaults
~~~~~~~~

- The default spatial finite-difference steps ``delta_X``, ``delta_Y``,
  ``delta_R`` and ``delta_Z`` are 1e-5 (v3: 1e-4). The wavevector steps are
  unchanged (0.1).
- ``interp_smoothing=0``, ``find_B_method="torbeam"`` and
  ``detailed_analysis_flag=True``, as in v3.
- ``poloidal_flux_zero_temperature`` defaults to ``None``, which is
  ``poloidal_flux_zero_density``.

Other interface changes
~~~~~~~~~~~~~~~~~~~~~~~

- ``output_path``, ``magnetic_data_path``, ``ne_data_path`` and
  ``Te_data_path`` accept a ``str`` or a ``pathlib.Path``. The output folder is
  created if it does not exist.
- A keyword argument that ``beam_me_up`` does not take is logged as a warning,
  with the closest argument name (``did you mean ...?``).
- The localisation is now called the filter function, in the outputs and the
  code: v3's ``loc_p``, ``loc_r``, ``loc_b``, ``loc_m``, ``loc_s`` are
  ``filter_p``, ``filter_r``, ``filter_b``, ``filter_m``,
  ``filter_s_<power>``, and there are ``filter_all_<power>`` and their path
  integrals ``int_filter_*_dL``.
- New argument: ``resonance_flag`` (#151). Temperature profiles are built when
  given.
- ``convert_hessians`` takes a required ``K_start`` (Cartesian K).
  ``analysis.py`` is its only caller.
- ``find_B_method="analytical"`` is ``CircularFluxSurfaceField``, as in v3; the
  analytical base class is ``CircularFieldBase``.
- New in ``hamiltonian.py`` and ``fun_general.py``: ``FIRST_ORDER_KEYS``,
  ``find_Booker_discriminant``.
- New outputs: ``int_filter_all_<power>_dL``, ``d_xhat_d_tau_dot_yhat``,
  ``g_hat_Cardano``, ``Psi_w`` on (``row_beam``, ``col_beam``) = (x, y), and
  the stacked ``H_Cardano``. ``filter_all_<power>`` is the array.
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
- ``CurvySlabField`` (``find_B_method="curvy_slab"``).
- The analytical fields ``CircularCrossSectionField`` and
  ``ConstantCurrentDensityField``, and ``find_B_method="unit-tests"``; use
  ``"analytical"``.
- ``check_Psi_output``, which warned on every run.
- ``open_*_npz``.
- The GUI (``GUI/scottypygui.py``) has not been ported to v4.
