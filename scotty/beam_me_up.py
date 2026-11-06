# -*- coding: utf-8 -*-
# Copyright 2018 - 2023, Valerian Hall-Chen and the Scotty contributors
# SPDX-License-Identifier: GPL-3.0

"""
The entry point of Scotty: `beam_me_up` traces the ray, propagates the beam
and analyses the result, saving everything to one output file.

Notes
~~~~~

- ``geometry`` is ``"cylindrical"`` (axisymmetric fields; :math:`K_\\zeta` is
  conserved, and :math:`K^2 = K_R^2 + (K_\\zeta / R)^2 + K_Z^2`) or
  ``"cartesian"`` (general 3D fields).
- ``launch_flag="vacuum_analytic"`` launches the beam in vacuum, propagates
  it analytically (as a Gaussian beam in vacuum) to the plasma boundary and
  applies the boundary condition there; the solvers start at the boundary.
  ``Psi_BC_flag`` is ``"continuous"`` (density continuous at the edge, but
  not its gradient), ``"discontinuous"`` (density jumps at the edge), or
  ``None`` (density continuous and flat at the edge, so :math:`K` and
  :math:`\\Psi` carry straight over from vacuum; this is checked, and an
  error suggests ``"continuous"`` if the profile is not zero with zero
  gradient at ``poloidal_flux_zero_density``). This is v3's
  ``vacuumLaunch_flag=True`` with ``vacuum_propagation_flag=True``.
- ``launch_flag="vacuum_numerical"`` launches the beam in vacuum and starts
  the solvers at the antenna, so the beam is integrated numerically through
  the vacuum, across the plasma boundary and into the plasma. There is no
  boundary condition (``Psi_BC_flag`` must be ``None``), and the quantities
  at the plasma entry (``Psi_3D_entry_labframe_*``,
  ``distance_from_launch_to_entry``) are NaN. This is v3's
  ``vacuumLaunch_flag=True`` with ``vacuum_propagation_flag=False``.
- ``launch_flag="plasma"`` starts the beam at ``launch_position``, which must
  be inside the plasma, with ``plasmaLaunch_K_cartesian`` (the wavevector,
  Cartesian lab-frame components :math:`K_X, K_Y, K_Z` in m⁻¹) and
  ``plasmaLaunch_Psi_3D_lab_cartesian`` (the 3×3 complex :math:`\\Psi` in the
  Cartesian lab frame, in m⁻²; not used with ``ray_tracing_flag``). There is
  no vacuum propagation or boundary condition, so the launch angles,
  ``launch_beam_width`` and ``launch_beam_curvature`` are ignored, and
  ``Psi_BC_flag`` has no effect. :math:`K` must satisfy the dispersion
  relation of the mode asked for (``mode_flag`` 1, -1, ``"O"`` or ``"X"``).
  With no launched beam, ``beam_waist``, ``filter_b`` and ``filter_all`` are NaN.
- Positions along the way: *launch* is where the beam starts in vacuum,
  *entry* is where it reaches the plasma boundary, and *initial* is where the
  solvers start: the first point in the plasma, or the antenna for
  ``launch_flag="vacuum_numerical"`` (e.g. ``q_launch``,
  ``Psi_3D_entry_labframe``, ``q_initial``).
- ``mode_flag`` picks a root of the Booker quadratic: 1 or -1 directly, or
  ``"O"`` / ``"X"``. The O-mode is the root more polarised along
  :math:`\\hat{b}` (for perpendicular propagation, :math:`E \\parallel B`);
  which of 1 and -1 that is depends on the plasma and :math:`K`. This also
  works at zero density (e.g. at the plasma edge), where both roots have
  :math:`N = 1`, from their polarisations' zero-density limits. It fails
  only for :math:`K` nearly along :math:`B`, where both modes are circularly
  polarised; give 1 or -1 there. The upper hybrid resonance stops the X-mode,
  whichever ``mode_flag`` it was launched with.
- ``ray_tracing_flag=True`` traces only the ray, with no beam.
- Scripts written for Scotty v3 can be converted with `scotty.compat`.

Output
~~~~~~

An `xarray.DataTree` with the groups ``inputs``, ``solver``, ``beam``, ``checks``
and ``diagnostics/dbs``, saved as ``scotty_output<suffix>.h5``.

Coordinates
~~~~~~~~~~~

- ``X, Y, Z`` - Lab Cartesian coordinates
- ``R, zeta, Z`` - Lab cylindrical coordinates
- ``x, y, g`` - Beam coordinates
- ``u1, u2, b`` - Field-aligned coordinates

Abbreviations
~~~~~~~~~~~~~

- ``bs`` - backscattered
- ``filter`` - filter function (``filter_b``, ``filter_p``, ``filter_r``,
  ``filter_m``, ``filter_s``)
- ``ne`` - equilibrium electron density
- ``dt`` - datatree (stores data)

Angles
~~~~~~

- ``theta`` - angle between g and u1, small when mismatch is small
- ``theta_m`` - mismatch angle, angle between u1 and K

Units
~~~~~

- SI units
- Distance in m
- Angles in rad
- Temperature in keV
- Electron cyclotron frequency positive
"""

import datetime
import logging
import numpy as np
import pathlib
from scotty.analysis import basic_analysis, dbs_analysis
from scotty.beam_solver import beam_tracing
from scotty.checks import (
    VALID_GEOMETRIES,
    VALID_LAUNCH_FLAGS,
    VALID_LAUNCH_MODE_FLAGS,
    VALID_PSI_BC_FLAGS,
    Parameters,
)
from scotty.data_io import (
    dict_to_dataset,
    inputs_to_dicts,
    solver_output_to_dicts,
)
from scotty.output_tree import to_output_tree
from scotty.geometry import (
    MagneticField_Cylindrical,
    MagneticField_Cartesian,
    create_magnetic_geometry,
)
from scotty.hamiltonian import initialise_hamiltonians, assign_hamiltonians
from scotty.launch import (
    find_plasma_entry_position,
    find_auto_delta_signs,
    find_O_mode_flag,
    find_plasma_entry_parameters,
)
from scotty.logger import config_logger, arr2str
from scotty.profile_fit import ProfileFitLike, profile_fit
from scotty.ray_solver import ray_tracing
from scotty.typing import FloatArray, PathLike
from scotty._version import __version__
from typing import Optional, Sequence, Tuple, Union, Literal, cast
import uuid
import xarray as xr


def beam_me_up(
    # Main parameters
    geometry: VALID_GEOMETRIES,
    poloidal_launch_angle_Torbeam: float,
    toroidal_launch_angle_Torbeam: float,
    launch_freq_GHz: float,
    launch_beam_width: float,
    launch_beam_curvature: float,
    launch_position: FloatArray,
    mode_flag: VALID_LAUNCH_MODE_FLAGS,
    # Solver settings and finite-difference parameters
    launch_flag: VALID_LAUNCH_FLAGS,
    Psi_BC_flag: VALID_PSI_BC_FLAGS,
    relativistic_flag: bool = False,  # includes relativistic corrections to electron mass when set to True
    resonance_flag: bool = True,  # stop the ray at a cyclotron (or, in X-mode, upper-hybrid) resonance
    auto_delta_sign: bool = True,  # For flipping signs to maintain forward difference
    delta_X: float = 1e-5,  # in the same units as data_X_coord
    delta_Y: float = 1e-5,  # in the same units as data_Y_coord
    delta_R: float = 1e-5,
    delta_Z: float = 1e-5,  # in the same units as data_Z_coord
    delta_K_X: float = 1e-1,  # in the same units as K_X
    delta_K_Y: float = 1e-1,  # in the same units as K_Y
    delta_K_R: float = 1e-1,
    delta_K_zeta: float = 1e-1,
    delta_K_Z: float = 1e-1,  # in the same units as K_Z
    len_tau: int = 4002,
    rtol: float = 1e-4,  # for solve_ivp of the ray/beam solvers
    atol: float = 1e-7,  # for solve_ivp of the ray/beam solvers
    poloidal_flux_enter: float = 1.0,
    poloidal_flux_zero_density: float = 1.0,  # When polflux >= poloidal_flux_zero_density, Scotty sets density = 0
    poloidal_flux_zero_temperature: Optional[
        float
    ] = None,  # Temperature analogue of poloidal_flux_zero_density; None: the same
    # Data input and output arguments
    magnetic_data_path: PathLike = pathlib.Path("."),
    ne_data_path: PathLike = pathlib.Path("."),
    Te_data_path: PathLike = pathlib.Path("."),
    input_filename_suffix: str = "",
    output_path: PathLike = pathlib.Path("."),
    output_filename_suffix: str = "",
    shot: Optional[int] = None,
    equil_time: Optional[Union[float, int]] = None,
    # Interpolation settings
    find_B_method: Union[
        str, MagneticField_Cylindrical, MagneticField_Cartesian
    ] = "torbeam",
    density_fit_method: Optional[Union[str, ProfileFitLike]] = None,
    density_fit_parameters: Optional[Sequence] = None,
    temperature_fit_method: Optional[Union[str, ProfileFitLike]] = None,
    temperature_fit_parameters: Optional[Sequence] = None,
    interp_order: Union[str, int] = 5,  # For the 3D interpolation functions
    interp_smoothing: int = 0,  # For the interpolation functions; 0 for no smoothing
    # Logging flags
    console_log_level: Union[
        str, int
    ] = "INFO",  # For returning log messages on console
    file_log_level: Optional[
        Union[str, int]
    ] = None,  # For returning log messages on a log file
    # Plotting flags
    figure_flag: bool = False,
    further_analysis_flag: bool = False,
    detailed_analysis_flag: bool = True,
    # Additional flags
    benchmarking_flag: bool = False,
    ray_tracing_flag: bool = False,
    return_dt_field: bool = False,  # For returning the datatree, field class
    # Keeping the extra kwargs for parsing later
    **kwargs,
) -> Union[
    xr.DataTree, Tuple[xr.DataTree, MagneticField_Cylindrical | MagneticField_Cartesian]
]:
    r"""Trace a beam through a plasma and compute its DBS filter function.

    Launches a Gaussian beam, traces its central ray and propagates the beam
    (its :math:`\Psi`) along it, then analyses the result, returning it as an
    `xarray.DataTree` and saving it to
    ``output_path / f"scotty_output{output_filename_suffix}.h5"``.

    Scripts written for Scotty v3 use arguments that v4 renamed or removed
    (e.g. ``vacuumLaunch_flag``, ``vacuum_propagation_flag``), and miss the
    required ``geometry``, ``launch_flag`` and ``Psi_BC_flag``. Convert them
    with ``beam_me_up(**scotty.compat.from_v3_kwargs(kwargs))``.

    Parameters
    ----------
    geometry : {"cylindrical", "cartesian"}
        Coordinates of the field and of the solvers: ``"cylindrical"``
        :math:`(R, \zeta, Z)` for axisymmetric fields, or ``"cartesian"``
        :math:`(X, Y, Z)` for general 3D fields. Outputs are given in both.
    poloidal_launch_angle_Torbeam : float
        Poloidal launch angle in degrees, in the TORBEAM convention
    toroidal_launch_angle_Torbeam : float
        Toroidal launch angle in degrees, in the TORBEAM convention. The
        antenna is at :math:`\zeta = 0` (:math:`Y = 0`)
    launch_freq_GHz : float
        Beam frequency in GHz
    launch_beam_width : float
        Beam width :math:`W` at the antenna, in m
    launch_beam_curvature : float
        Curvature of the beam's wavefronts at the antenna, :math:`1/R_c`, in
        m⁻¹ (negative for a converging beam)
    launch_position : array_like, shape (3,)
        Antenna position: :math:`(R, \zeta, Z)` for ``geometry="cylindrical"``
        (with :math:`\zeta = 0`), :math:`(X, Y, Z)` for ``"cartesian"``. For
        ``launch_flag="plasma"``, where the solvers start, inside the plasma
    mode_flag : {1, -1, "O", "X"}
        Which root of the Booker quadratic to trace: 1 or -1 directly, or
        ``"O"`` / ``"X"``, decided from the two roots' polarisations where the
        solvers start (see the module notes). ``"O"`` and ``"X"`` raise where
        the polarisations are too alike to tell apart (:math:`K` nearly along
        :math:`B`)
    launch_flag : {"vacuum_analytic", "vacuum_numerical", "plasma"}
        How the beam reaches the plasma:

        - ``"vacuum_analytic"``: launched at the antenna, propagated
          analytically through the vacuum to the plasma boundary
          (``poloidal_flux_enter``), where ``Psi_BC_flag`` is applied; the
          solvers start there. v3's ``vacuumLaunch_flag=True``,
          ``vacuum_propagation_flag=True``.
        - ``"vacuum_numerical"``: launched at the antenna, where the solvers
          start; the beam is integrated numerically through the vacuum and
          into the plasma, with no boundary conditions. The plasma-entry
          quantities are saved as NaN. v3's ``vacuumLaunch_flag=True``,
          ``vacuum_propagation_flag=False``.
        - ``"plasma"``: the solvers start at ``launch_position``, inside the
          plasma, with ``plasmaLaunch_K_cartesian`` and
          ``plasmaLaunch_Psi_3D_lab_cartesian``. v3's
          ``vacuumLaunch_flag=False``.
    Psi_BC_flag : {"continuous", "discontinuous", None}
        Boundary conditions for :math:`K` and :math:`\Psi` at the plasma
        boundary, used with ``launch_flag="vacuum_analytic"`` only:

        - ``"continuous"``: the density is continuous at the boundary, but
          its gradient is not.
        - ``"discontinuous"``: the density jumps at the boundary.
        - ``None``: no boundary conditions. :math:`K` and :math:`\Psi`
          carry straight over from vacuum, which needs the density to be zero
          with zero gradient at the boundary; this is checked.

        Must be ``None`` for ``launch_flag="vacuum_numerical"`` (there is no
        boundary step), and has no effect for ``"plasma"``. Booleans (v3's
        deprecated values) raise a `ValueError`.
    relativistic_flag : bool, default False
        Include the relativistic correction to the electron mass, from the
        temperature profile
    resonance_flag : bool, default True
        Stop the ray at the fundamental or second-harmonic electron cyclotron
        resonance, and (for the X-mode) the upper hybrid resonance
    auto_delta_sign : bool, default True
        Flip the signs of the spatial steps so that the forward differences
        point into the plasma at the plasma entry point
    delta_X, delta_Y, delta_R, delta_Z : float, default 1e-5
        Spatial finite-difference steps, in m: ``delta_X`` and ``delta_Y`` for
        ``geometry="cartesian"``, ``delta_R`` for ``"cylindrical"``,
        ``delta_Z`` for both. v3's default was 1e-4
    delta_K_X, delta_K_Y, delta_K_R, delta_K_zeta, delta_K_Z : float, default 0.1
        Wavevector finite-difference steps, in m⁻¹ (``delta_K_zeta`` is a
        step in the toroidal mode number)
    len_tau : int, default 4002
        Number of points along the ray in the output. The filter function is
        integrated on these points. Where the beam focuses sharply at the
        cut-off, its peak can be narrower than their spacing if there are far
        fewer of them. v3's default was 102
    rtol, atol : float, default 1e-4, 1e-7
        Relative and absolute tolerances of the ray and beam solvers
        (`scipy.integrate.solve_ivp`). Use tighter ones (e.g. 1e-6, 1e-9) for
        anything read close to the cut-off, where :math:`\Psi` is steep, and
        when :math:`K` at the cut-off is below about a tenth of its vacuum
        value. v3's defaults were 1e-3, 1e-6
    poloidal_flux_enter : float, default 1.0
        Normalised poloidal flux of the plasma boundary: where the beam
        enters the plasma, and where the ray is stopped when it leaves
    poloidal_flux_zero_density : float, default 1.0
        At and above this normalised poloidal flux, the electron density is
        zero. Must be at least ``poloidal_flux_enter``
    poloidal_flux_zero_temperature : float, optional
        Temperature analogue of ``poloidal_flux_zero_density``; ``None`` (the
        default) uses ``poloidal_flux_zero_density``. If given, must be at
        least ``poloidal_flux_enter``
    magnetic_data_path, ne_data_path, Te_data_path : str or pathlib.Path, default "."
        Directories of the magnetic field, electron density and electron
        temperature files
    input_filename_suffix : str, default ""
        Suffix of the input file names (e.g. ``ne{suffix}.dat``)
    output_path : str or pathlib.Path, default "."
        Directory for the output file (and the log file); created if it does
        not exist
    output_filename_suffix : str, default ""
        Suffix of the output file name, ``scotty_output{suffix}.h5``
    shot : int, optional
        Shot number, for the field files that need one
    equil_time : float, optional
        Time of the equilibrium, for the field files that hold several
    find_B_method : str or MagneticField_Cylindrical or MagneticField_Cartesian, default "torbeam"
        Source of the magnetic field: a field object, or the name of a file
        format (``"torbeam"``, ``"omfit"``, ``"test"``, ``"test_notime"``,
        ``"UDA_saved"``, ``"EFITpp"``) or of an analytical field
        (``"analytical"``: circular flux surfaces, which needs ``B_T_axis``,
        ``R_axis``, ``minor_radius_a`` and ``B_p_a``)
    density_fit_method : str or callable, optional
        Electron density profile (in 10¹⁹ m⁻³) as a function of the normalised
        poloidal flux: a callable (e.g. a `scotty.profile_fit.ProfileFit`),
        or the name of a fit, with ``density_fit_parameters``. ``None`` or
        ``"smoothing-spline-file"`` with no parameters reads
        ``ne{input_filename_suffix}.dat`` from ``ne_data_path``
    density_fit_parameters : sequence, optional
        Parameters of the named density fit
    temperature_fit_method : str or callable, optional
        Electron temperature profile (in keV), as ``density_fit_method``;
        used only with ``relativistic_flag``
    temperature_fit_parameters : sequence, optional
        Parameters of the named temperature fit
    interp_order : int or str, default 5
        Order of the field and profile interpolation: 1, 3, 5 or
        ``"linear"``, ``"cubic"``, ``"quintic"``
    interp_smoothing : int, default 0
        Smoothing of the interpolating splines; 0 for none
    console_log_level : str or int, default "INFO"
        Level of the log messages printed to the console
    file_log_level : str or int, optional
        Level of the log messages written to a log file in ``output_path``;
        ``None`` for no log file
    figure_flag : bool, default False
        Stored in the output's inputs. v4 does not make figures here, so
        ``True`` only logs a warning; see `scotty.plotting`
    further_analysis_flag : bool, default False
        Stored in the output's inputs
    detailed_analysis_flag : bool, default True
        Stored in the output's inputs
    benchmarking_flag : bool, default False
        Stored in the output's inputs
    ray_tracing_flag : bool, default False
        Trace only the ray, with no beam (v3's ``quick_run``; see
        `scotty.compat.quick_cutoff`)
    return_dt_field : bool, default False
        Also return the magnetic field object
    **kwargs
        ``B_T_axis``, ``R_axis``, ``minor_radius_a`` and ``B_p_a`` for
        ``find_B_method="analytical"``; ``plasmaLaunch_K_cartesian`` (shape
        (3,), m⁻¹) and ``plasmaLaunch_Psi_3D_lab_cartesian`` (shape (3, 3),
        m⁻², complex), both in Cartesian lab-frame components, for
        ``launch_flag="plasma"``; and ``create_magnetic_geometry``, a dict of
        extra arguments for the field. Other keyword arguments are ignored,
        with a warning that suggests the closest argument name.

    Returns
    -------
    xarray.DataTree, or tuple of (xarray.DataTree, field)
        The output, with the groups ``inputs``, ``solver``, ``beam``, ``checks``
        and ``diagnostics/dbs``, and the field if ``return_dt_field``
    """

    ##################################################
    #
    # INITIALISATION ROUTINE
    #
    ##################################################
    params = Parameters(
        # Main parameters
        geometry=geometry,
        poloidal_launch_angle_Torbeam=poloidal_launch_angle_Torbeam,
        toroidal_launch_angle_Torbeam=toroidal_launch_angle_Torbeam,
        launch_freq_GHz=launch_freq_GHz,
        launch_beam_width=launch_beam_width,
        launch_beam_curvature=launch_beam_curvature,
        q_launch=launch_position,
        mode_flag_launch=mode_flag,
        # Solver settings and finite-difference parameters
        launch_flag=launch_flag,
        Psi_BC_flag=Psi_BC_flag,
        relativistic_flag=relativistic_flag,
        resonance_flag=resonance_flag,
        auto_delta_sign=auto_delta_sign,
        deltas_cylindrical=np.array(
            [delta_R, delta_Z, delta_K_R, delta_K_zeta, delta_K_Z], dtype=float
        ),
        deltas_cartesian=np.array(
            [delta_X, delta_Y, delta_Z, delta_K_X, delta_K_Y, delta_K_Z], dtype=float
        ),
        len_tau=len_tau,
        rtol=rtol,
        atol=atol,
        poloidal_flux_enter=poloidal_flux_enter,
        poloidal_flux_zero_density=poloidal_flux_zero_density,
        poloidal_flux_zero_temperature=poloidal_flux_zero_temperature,
        # Data input and output arguments
        magnetic_data_path=magnetic_data_path,
        ne_data_path=ne_data_path,
        Te_data_path=Te_data_path,
        input_filename_suffix=input_filename_suffix,
        output_path=output_path,
        output_filename_suffix=output_filename_suffix,
        shot=shot,
        equil_time=equil_time,
        # Interpolation settings
        find_B_method=find_B_method,
        density_fit_method=density_fit_method,
        density_fit_parameters=density_fit_parameters,
        temperature_fit_method=temperature_fit_method,
        temperature_fit_parameters=temperature_fit_parameters,
        interp_order=interp_order,
        interp_smoothing=interp_smoothing,
        # Plotting flags
        figure_flag=figure_flag,
        further_analysis_flag=further_analysis_flag,
        detailed_analysis_flag=detailed_analysis_flag,
        # Additional flags
        benchmarking_flag=benchmarking_flag,
        ray_tracing_flag=ray_tracing_flag,
        # return_dt_field = return_dt_field,
        # Extra kwargs for parsing
        **kwargs,
    )

    # Setting up the logger
    config_logger(
        console_log_level,
        file_log_level,
        params.output_path,
        params.output_filename_suffix,
    )
    log = logging.getLogger(__name__)
    log.debug(f"Saved and validated launch parameters")
    log.debug(f"Initialised logger")
    if figure_flag:
        log.warning(
            "figure_flag=True makes no figures in v4: use scotty.plotting instead"
        )

    log.info(f"""\n
    ##################################################
    #
    # STARTING ROUTINE
    #
    ##################################################
    #
    # Beam trace me up, Scotty!
    # Version {__version__}
    # Run ID: {uuid.uuid4()}
    #
    ##################################################
    #
    # Starting run for:
    #   - Geometry = {params.geometry}
    #   - Pol. launch angle = {params.poloidal_launch_angle_deg_Torbeam} deg
    #   - Tor. launch angle = {params.toroidal_launch_angle_deg_Torbeam} deg
    #   - Launch frequency = {params.launch_frequency_GHz} GHz
    #   - Launch beam width = {params.launch_beam_width} m
    #   - Launch beam curvature = {params.launch_beam_curvature} m^-1
    #   - Launch position {"[R,zeta,Z]" if params.geometry == "cylindrical" else "[X,Y,Z]"} = {arr2str(params.q_launch)}
    #   - Mode flag = {params.mode_flag_launch}
    #
    #   - find_B_method = {params.find_B_method}
    #   - launch_flag = {params.launch_flag}
    #   - Psi_BC_flag = {params.Psi_BC_flag}
    #   - relativistic flag = {params.relativistic_flag}
    #   - figure flag = {params.figure_flag}
    #   - further_analysis flag = {params.further_analysis_flag}
    #   - detailed_analysis flag = {params.detailed_analysis_flag}
    #
    ##################################################
    """)

    log.debug(f"Setting experimental profiles for density")
    params.set_experimental_profiles()

    # UUsing the electron density data, create a spline fit as a
    # function of poloidal flux, i.e.:
    # density_fit(poloidal_flux) = electron_density
    density_fit = make_fit(
        method=params.density_fit_method,
        poloidal_flux_zero_density=params.poloidal_flux_zero_density,
        parameters=params.density_fit_parameters,
        filename=params.ne_filename,
    )

    # Using the temperature data, create a fit as a function of poloidal
    # flux, i.e. temperature_fit(poloidal_flux) = temperature. Only used for
    # relativistic corrections, as in v3 (beam-tracing/Scotty#151)
    if params.relativistic_flag:
        temperature_fit = make_fit(
            method=params.temperature_fit_method,
            poloidal_flux_zero_density=params.poloidal_flux_zero_temperature,
            parameters=params.temperature_fit_parameters,
            filename=params.Te_filename,
            quantity="temperature",
        )
    else:
        temperature_fit = None

    # Analytical field parameters can be passed directly to beam_me_up (as in
    # v3) or nested in `create_magnetic_geometry`; the nested ones take priority
    field_kwargs = {
        key: value
        for key, value in {
            "B_T_axis": params.B_T_axis,
            "R_axis": params.R_axis,
            "minor_radius_a": params.minor_radius_a,
            "B_p_a": params.B_p_a,
        }.items()
        if value is not None
    }
    field_kwargs.update(kwargs.get("create_magnetic_geometry", {}))

    # If the user has already pre-loaded a field, use it
    # Otherwise, create a new one from the specified data pathway
    field = create_magnetic_geometry(
        geometry=params.geometry,
        find_B_method=find_B_method,  # not from params because we only store the string/name, but find_B_method is probably already a MagField class
        interp_order_str=params.interp_order_magnetic_data_str,
        interp_order_int=params.interp_order_magnetic_data_int,
        interp_smoothing=params.interp_smoothing,
        magnetic_data_path=params.magnetic_data_path,
        input_filename_suffix=params.input_filename_suffix,
        shot=params.shot,
        equil_time=params.equil_time,
        **field_kwargs,
    )

    log.info(f"""\n
    ##################################################
    #
    # LAUNCH ROUTINE
    #
    ##################################################
    """)

    if params.launch_flag == "plasma":
        log.warning(
            "`launch_flag` = 'plasma': the beam starts at `launch_position` with "
            "`plasmaLaunch_K_cartesian` and `plasmaLaunch_Psi_3D_lab_cartesian`, so "
            "the launch angles, `launch_beam_width` and `launch_beam_curvature` are "
            "not used"
        )

    if params.launch_flag == "vacuum_numerical":
        log.info(
            "`launch_flag` = 'vacuum_numerical': the solvers start at the antenna and "
            "integrate through the vacuum into the plasma, with no boundary conditions"
        )

    def _entry_position():
        return find_plasma_entry_position(
            poloidal_launch_angle_deg_Torbeam=params.poloidal_launch_angle_deg_Torbeam,
            toroidal_launch_angle_deg_Torbeam=params.toroidal_launch_angle_deg_Torbeam,
            q_launch=params.q_launch,
            launch_flag=params.launch_flag,
            field=field,
            poloidal_flux_enter=params.poloidal_flux_enter,
            boundary_adjust=1e-6,
        )

    # Where the solvers start: the plasma entry position (or the launch
    # position, inside the plasma, for a plasma launch), or the antenna for a
    # numerical vacuum launch. The delta signs are set at the entry position
    # in every case, as in v3, so that the forward differences point into the
    # plasma where the beam crosses its boundary; a numerical vacuum launch
    # looks for the entry only for this, as v3 did
    if params.launch_flag == "vacuum_numerical":
        params.q_initial = params.q_launch
        q_delta_signs = _entry_position() if params.auto_delta_sign else params.q_launch
    else:
        params.q_initial = _entry_position()
        q_delta_signs = params.q_initial

    # Setting the delta signs
    spatial_deltas = find_auto_delta_signs(
        auto_delta_sign=params.auto_delta_sign,
        q_initial=q_delta_signs,
        deltas=params.deltas[:-3],
        field=field,
    )

    if params.geometry == "cylindrical":
        params.delta_R, params.delta_Z = spatial_deltas
    elif params.geometry == "cartesian":
        params.delta_X, params.delta_Y, params.delta_Z = spatial_deltas
    # The Hamiltonians below take their spacings from `params.deltas`, so
    # update its spatial part as well
    params.deltas = np.concatenate([spatial_deltas, params.deltas[-3:]])

    # Initialises the Hamiltonian H for `mode_flag`s +1 and -1
    (
        hamiltonian_pos1,
        hamiltonian_neg1,
    ) = initialise_hamiltonians(
        launch_angular_freq=params.launch_angular_frequency,
        deltas=params.deltas,
        field=field,
        density_fit=density_fit,
        temperature_fit=temperature_fit,
    )

    # Calculating the plasma entry parameters
    (
        params.K_launch,
        params.K_initial,
        params.Psi_3D_launch_labframe,
        params.Psi_3D_entry_labframe,
        params.Psi_3D_initial_labframe,
        params.distance_from_launch_to_entry,
        params.e_hat_initial,
        params.mode_flag_initial,
        params.mode_index,
    ) = find_plasma_entry_parameters(
        ray_tracing_flag=params.ray_tracing_flag,
        launch_flag=params.launch_flag,
        Psi_BC_flag=params.Psi_BC_flag,
        mode_flag_launch=params.mode_flag_launch,
        poloidal_launch_angle_deg_Torbeam=params.poloidal_launch_angle_deg_Torbeam,
        toroidal_launch_angle_deg_Torbeam=params.toroidal_launch_angle_deg_Torbeam,
        q_launch=params.q_launch,
        q_initial=params.q_initial,
        launch_beam_width=params.launch_beam_width,
        launch_beam_curvature=params.launch_beam_curvature,
        field=field,
        K_plasmaLaunch_cartesian=params.K_plasmaLaunch_cartesian,
        Psi_3D_plasmaLaunch_labframe_cartesian=params.Psi_3D_plasmaLaunch_labframe_cartesian,
        poloidal_flux_zero_density=params.poloidal_flux_zero_density,
        hamiltonian_pos1=hamiltonian_pos1,
        hamiltonian_neg1=hamiltonian_neg1,
        tol_H=1e-2,
        tol_O_mode_polarisation=0.25,
    )

    # Assigning the correct Hamiltonian
    (
        hamiltonian,
        hamiltonian_other,
    ) = assign_hamiltonians(
        mode_flag_initial=params.mode_flag_initial,
        hamiltonian_pos1=hamiltonian_pos1,
        hamiltonian_neg1=hamiltonian_neg1,
        q_initial=params.q_initial,
        K_initial=params.K_initial,
        tol_H=1e-5,
    )

    # Whether the traced root is the X-mode, from the two roots'
    # polarisations where the solvers start, for the upper hybrid resonance
    try:
        X_mode = params.mode_flag_initial != find_O_mode_flag(
            params.q_initial,
            params.K_initial,
            field,
            params.launch_angular_frequency,
            density_fit,
            temperature_fit,
        )
    except ValueError as error:
        # K nearly along B: the roots' polarisations are too alike to label
        X_mode = params.mode_flag_initial == -1
        log.warning(
            f"{error}. For the upper hybrid resonance, taking `mode_flag` = -1 as "
            f"the X-mode, so this is the {'X' if X_mode else 'O'}-mode"
        )
    log.info(
        f"Tracing the {'X' if X_mode else 'O'}-mode (`mode_flag` = {params.mode_flag_initial})"
    )

    log.info(f"""\n
    ##################################################
    #
    # RAY TRACING ROUTINE
    #
    ##################################################
    """)

    ray_tracing_result = ray_tracing(
        q_initial=params.q_initial,
        K_initial=params.K_initial,
        poloidal_flux_enter=params.poloidal_flux_enter,
        hamiltonian=hamiltonian,
        ray_tracing_flag=params.ray_tracing_flag,
        rtol=params.rtol,
        atol=params.atol,
        len_tau=params.len_tau,
        resonance_flag=params.resonance_flag,
        X_mode=X_mode,
        # tau_max = 1e5,
    )

    if (
        params.ray_tracing_flag
    ):  # if ray_tracing_flag, then ray_tracing_result is 2-tuple of (tau_arr, q_K_arrs)
        (
            params.solver_status,
            params.solver_nfev,
            params.solver_duration,
            params.tau_output,
            q_K_output,
        ) = cast(
            Tuple[Literal[1, 0, -1], int, float, FloatArray, FloatArray],
            ray_tracing_result,
        )
        params.q_output = q_K_output[:3].T  # (N, 3)
        params.K_output = q_K_output[3:].T  # (N,3 )
        params.Psi_3D_output_labframe = None
    else:  # otherwise just take the tau_array with tau of terminating event and pass into beam-tracing
        _, _, _, tau_points, tau_terminating_event = cast(
            Tuple[Literal[1, 0, -1], int, float, FloatArray, float], ray_tracing_result
        )

        log.info(f"""\n
    ##################################################
    #
    # BEAM TRACING ROUTINE
    #
    ##################################################
    """)

        (
            params.solver_status,
            params.solver_nfev,
            params.solver_duration,
            params.tau_output,  # (N,)
            params.q_output,  # (N, 3)
            params.K_output,  # (N, 3)
            params.Psi_3D_output_labframe,  # (N, 3, 3)
        ) = beam_tracing(
            tau_leave=tau_terminating_event,
            tau_points=tau_points,
            q_initial=params.q_initial,
            K_initial=params.K_initial,
            Psi_3D_initial_labframe=params.Psi_3D_initial_labframe,
            hamiltonian=hamiltonian,
            rtol=params.rtol,
            atol=params.atol,
        )

    # Coordinate conversions for later
    params.coordinate_conversions()

    # Once the ray/beam-tracing is complete, save the data from params
    # as xarray datasets
    inputs_ds = dict_to_dataset(*inputs_to_dicts(params, field))
    solver_output_ds = dict_to_dataset(*solver_output_to_dicts(params))

    # Then do basic analysis (i.e. no DBS) and also DBS analysis and
    # convert both to xarray datasets
    analysis_ds = dict_to_dataset(
        *basic_analysis(
            inputs=inputs_ds,
            solver_output=solver_output_ds,
            hamiltonian=hamiltonian,
            hamiltonian_other=hamiltonian_other,
            field=field,
            density_fit=density_fit,
            temperature_fit=temperature_fit,
        )
    )

    dbs_analysis_ds = dict_to_dataset(
        *dbs_analysis(
            inputs=inputs_ds,
            solver_output=solver_output_ds,
            analysis=analysis_ds,
            hamiltonian=hamiltonian,
            hamiltonian_other=hamiltonian_other,
            field=field,
            density_fit=density_fit,
            temperature_fit=temperature_fit,
        )
    )

    dt = to_output_tree(inputs_ds, solver_output_ds, analysis_ds, dbs_analysis_ds)
    dt.attrs = {
        "title": output_filename_suffix,
        "software_name": "scotty-beam-tracing",
        "software_version": __version__,
        "date_created": str(datetime.datetime.now()),
        # "id": str(run_id),
    }
    dt.to_netcdf(
        filepath=params.output_path / f"scotty_output{output_filename_suffix}.h5",
        engine="h5netcdf",
        invalid_netcdf=True,
    )

    if return_dt_field:
        return dt, field
    else:
        return dt


def make_fit(
    method: Optional[Union[str, ProfileFitLike]],
    poloidal_flux_zero_density: float,
    parameters: Optional[Sequence],
    filename: Optional[PathLike],
    quantity: str = "density",
) -> ProfileFitLike:
    """Profile fit for the electron density or temperature (``quantity``).
    ``poloidal_flux_zero_density`` is the flux beyond which the profile is
    zero, for whichever quantity it is"""

    log = logging.getLogger(__name__)
    log.debug(f"Making {quantity} fit profile")

    if callable(method):
        return method

    if not isinstance(method, (str, type(None))):
        raise TypeError(
            f"Unexpected method type. Expected callable, str, or None, but got '{type(method)}'"
        )

    if parameters is None:
        raise ValueError(
            f"Passing `{quantity}_fit_method` ({method}) as string or None requires a list or array of parameters"
        )

    return profile_fit(method, poloidal_flux_zero_density, parameters, filename)
