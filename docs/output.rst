.. _output:

Output parameters
==========================
Saved in data_input

* launch_K - initial wavevector
* launch_position - initial position
* mode_flag - polarisation

See `beam_me_up() <scotty.beam_me_up.beam_me_up>` for a detailed
description of input parameters.

Saved in data_output. Everything in cylindrical units unless otherwise stated ('Cartesian' or with the lower case 'x' or 'y')

* q_R_array - beam trajectory, R coordinate
* q_zeta_array - beam trajectory, zeta coordinate
* q_Z_array - beam trajectory, Z coordinate

Saved in analysis_output. Everything in cylindrical units unless otherwise stated ('Cartesian' or with the lower case 'x' or 'y')

* Psi_xx_output - xx component of Psi_w
* Psi_xy_output - xy component of Psi_w
* Psi_yy_output - yy component of Psi_w
