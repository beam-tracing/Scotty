.. _naming:

Names
=====

This page gives the rules for the names of Scotty's arguments, outputs and
variables, so that one quantity has one name everywhere. The examples come
from the output of `beam_me_up() <scotty.beam_me_up.beam_me_up>`. Some
public names do not follow the rules yet: v4.0.0rc1 froze the interface, so
they stay until a later release. The last section lists them.

Rules
-----

.. list-table::
   :header-rows: 1
   :widths: 18 42 40

   * - What
     - Rule
     - Examples
   * - Magnitude of a vector
     - ``<vector>_magnitude``
     - ``B_magnitude``, ``g_magnitude``, ``K_output_magnitude``
   * - Unit vector
     - ``<vector>_hat``, also inside longer names
     - ``b_hat_cartesian``, ``g_hat_cartesian``, ``normal_hat``
   * - Coordinate system
     - ``_cartesian`` or ``_cylindrical`` on each vector and matrix in that
       system. A vector in another frame names the frame in its dimension:
       ``col_field`` (:math:`u_1, u_2, \hat{b}`) or ``col_beam``
       (:math:`\hat{x}, \hat{y}`)
     - ``q_output_cartesian``, ``K_initial_cylindrical``; ``e_hat``
       (``col_field``), ``Psi_w`` (``col_beam``)
   * - Dimensions
     - Short forms ``_cart`` and ``_cyld``
     - ``row_cart``, ``col_cart``, ``col_cyld``
   * - Components
     - ``_R``, ``_zeta``, ``_Z`` in cylindrical coordinates; ``_X``, ``_Y``,
       ``_Z`` in Cartesian coordinates
     - ``B_X``, ``g_X_Cardano``
   * - Order of the parts
     - ``<quantity>_<place>_<component or system>``
     - ``q_launch_cartesian``, ``K_output_magnitude``
   * - Place along the ray
     - ``_launch``: where the beam starts. ``_entry``: at the plasma
       boundary, before the boundary conditions (``Psi_BC_flag``).
       ``_initial``: at the plasma boundary after the boundary conditions,
       where the solvers start. ``_output``: what the solvers return, along
       :math:`\tau`
     - ``q_launch_cartesian``, ``Psi_3D_entry_labframe_cartesian``,
       ``Psi_3D_initial_labframe_cartesian``, ``q_output_cartesian``
   * - Derivative
     - ``d_<A>_d_<B>``, because A often contains ``_``
     - ``d_theta_d_tau``
   * - Gradient
     - ``grad_`` with respect to the position, ``gradK_`` with respect to the
       wavevector
     - ``grad_grad_H_cartesian``, ``gradK_grad_H_cartesian``,
       ``gradK_gradK_H_cartesian``
   * - Poloidal flux
     - ``poloidal_flux``
     - ``poloidal_flux_enter``, ``poloidal_flux_zero_density``
   * - Words
     - Full words, apart from the symbols in the next section. British
       spelling: "normalised", "polarisation", "metres"
     - ``launch_angular_frequency``, ``normalised_omega_pe``
   * - Form of the dispersion relation
     - ``_Booker`` for the Booker quartic; ``_Cardano`` for the eigenvalues of
       the dispersion tensor (Cardano's formula)
     - ``H_Booker``, ``H_Cardano``, ``g_magnitude_Cardano``
   * - Tolerance
     - ``tol_<quantity>``, as ``rtol`` and ``atol``
     - ``tol_H``
   * - Names of people and codes
     - People with a capital letter; codes in lower case
     - ``_Booker``, ``_Cardano``, ``find_Rayleigh_length``;
       ``find_B_method="torbeam"``
   * - Options
     - ``<name>_flag``: ``True`` or ``False``, a name, or +1 or -1
     - ``analysis_flag``, ``launch_flag``, ``mode_flag``
   * - Valid names
     - Each output name is a Python identifier, so that ``ds.<name>`` works
     - ``k_perp1_bs``

Symbols
-------

These symbols are names on their own and in longer names.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Symbol
     - Meaning
   * - ``q``
     - The position of the ray
   * - ``K``
     - The wavevector. In cylindrical coordinates, ``K_zeta`` is the toroidal
       mode number, not the toroidal component of :math:`K`
   * - ``Psi``
     - The beam's :math:`\Psi`: ``Psi_3D_*_labframe_*`` is the 3 x 3 matrix in
       the lab frame, and ``Psi_w`` is the 2 x 2 matrix in the beam frame
   * - ``B``, ``b_hat``
     - The magnetic field and its unit vector
   * - ``g``
     - The group velocity, :math:`dq/d\tau`
   * - ``H``
     - The dispersion relation (the Hamiltonian). ``H_Booker_other`` is its
       value for the other mode
   * - ``tau``
     - The beam parameter :math:`\tau`, along which the solvers integrate
   * - ``theta``
     - The angle between the beam frame and the frame of the magnetic field
       (Hall-Chen, Parra and Hillesheim, 2022; see :ref:`references`)
   * - ``theta_m``
     - The mismatch angle: the angle between :math:`K` and the plane
       perpendicular to the magnetic field
   * - ``ne``, ``Te``
     - The electron density and temperature, in the names of data files and
       spline settings: ``ne_data_path``, ``interp_order_Te``
   * - ``R``, ``zeta``, ``Z``, ``X``, ``Y``
     - The coordinates

Beam frame
----------

``g_hat`` is along the group velocity. ``y_hat`` is perpendicular to
``b_hat`` and ``g_hat``, and ``x_hat`` completes the frame. ``Psi_xx``,
``Psi_xy``, ``Psi_yy``, ``Psi_xg``, ``Psi_yg`` and ``Psi_gg`` are the
components of :math:`\Psi` in this frame. ``width_1`` and ``width_2`` (and
``curvature_1`` and ``curvature_2``) are along the principal axes of
:math:`\Psi_w`, with their directions in ``width_1_direction`` and
``width_2_direction``.

Doppler backscattering
----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Name
     - Meaning
   * - ``cutoff_index``
     - The index of the point where :math:`|K|` is smallest.
       ``beam_path_length_wrt_cutoff`` is the distance along the beam from
       that point
   * - ``_bs``
     - The wavenumber that the Bragg condition selects for backscattering:
       ``k_perp1_bs``, ``k_perp2_bs``
   * - ``filterfun_p``, ``filterfun_r``, ``filterfun_b``, ``filterfun_m``
     - The polarisation, ray, beam and mismatch pieces of the filter
       function. ``filterfun_all`` is their product
   * - ``int_<name>_dl``
     - The integral of ``<name>`` along the beam path length :math:`l`:
       ``int_filterfun_all_dl``
   * - ``delta_ne_sq_<p>``
     - A model turbulence spectrum with the power law ``<p>``:
       ``delta_ne_sq_-10_3`` is :math:`k^{-10/3}`

Names that do not follow the rules yet
--------------------------------------

v4.0.0rc1 froze these public names, so they stay until a later release.
Internal variables that hold the same quantity keep the public name, so that
one quantity has one name in the code.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Rule
     - Names
   * - Magnitude
     - The parameter ``B_total`` of 8 functions in `scotty.fun_general`:
       ``find_normalised_gyro_ang_freq``, ``find_normalised_angular_freqs``,
       ``find_epsilon_terms``, ``find_Booker_terms``,
       ``find_Booker_discriminant``, ``find_Booker_alpha``,
       ``find_Booker_beta`` and ``find_Booker_gamma``. `scotty.hamiltonian`
       and `scotty.launch` use ``B_magnitude``
   * - Unit vector
     - ``d_xhat_d_tau``, ``d_ghat_d_tau``, ``d_xhat_d_tau_dot_yhat``,
       ``kappa_dot_xhat``, ``kappa_dot_yhat``, ``kappa_dot_ghat``,
       ``dbhat_dX``, ``dbhat_dY``, ``dbhat_dZ``, ``grad_bhat_cartesian``,
       ``xhat_dot_grad_bhat_dot_*``, ``yhat_dot_grad_bhat_dot_*``
   * - Coordinate system
     - ``normal_hat``, ``binormal_hat`` and ``k_perp1_hat`` are Cartesian
       but have no ``_cartesian``
   * - Order of the parts
     - ``q_R_output``, ``q_zeta_output``, ``q_Z_output``, ``K_R_output``,
       ``K_zeta_output``, ``K_Z_output``
   * - Derivative
     - ``dK_dtau_cartesian``, ``dbhat_dX``, ``dpolflux_dX`` (and ``_dY``,
       ``_dZ``); the keys of `Hamiltonian.derivatives()
       <scotty.hamiltonian.Hamiltonian.derivatives>` (``"dH_dR"``, ...)
   * - Poloidal flux
     - The output ``polflux`` and ``dpolflux_d*``; the field methods
       ``polflux``, ``polflux_cartesian`` and ``d_polflux_d*``
   * - Words
     - ``launch_freq_GHz``; the parameters ``launch_angular_freq`` and
       ``freq_GHz``; the functions ``find_normalised_plasma_ang_freq``,
       ``find_normalised_gyro_ang_freq``, ``find_normalised_angular_freqs``,
       ``freq_GHz_to_angular_frequency`` and ``freq_GHz_to_wavenumber``
   * - Tolerance
     - ``H_tol`` and ``fraction_tol`` (`scotty.checks`)
   * - Names of codes
     - ``poloidal_launch_angle_Torbeam``, ``toroidal_launch_angle_Torbeam``
   * - Valid names
     - ``delta_ne_sq_-10_3``, ``delta_ne_sq_-13_3`` and their integrals
       ``int_*_dl``: read them with ``ds["delta_ne_sq_-10_3"]``
