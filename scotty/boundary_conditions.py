import logging
import numpy as np
from scipy.optimize import newton
from scotty.checks import (
    VALID_BOUNDARY_FLAGS,
    VALID_FIELDS,
    MagneticField_Cylindrical,
    MagneticField_Cartesian,
)
from scotty.fun_general import (
    angular_frequency_to_wavenumber,
    find_normalised_angular_freqs,
    find_q_labframe_cyl_to_cart,
    find_vector_and_q_cyl_to_cart,
    find_q_labframe_cart_to_cyl,
    find_K_labframe_cart_to_cyl,
    find_K_labframe_cyl_to_cart,
    find_Psi_3D_labframe_cart_to_cyl,
    find_Psi_3D_labframe_cyl_to_cart,
    find_Booker_discriminant,
    find_Booker_terms,
)
from scotty.hamiltonian import Hamiltonian
from scotty.logger import arr2str
from scotty.typing import ArrayLike, FloatArray, ComplexFloatArray
from typing import Tuple, Literal, Optional, cast

log = logging.getLogger(__name__)

##################################################
#
# BOUNDARY CONDITION CHECKS
#
##################################################


def check_vector_pointing_into_plasma(
    q: FloatArray, vector: FloatArray, field: VALID_FIELDS
) -> None:
    if not q.shape == (3,):
        raise ValueError(f"`q` must have shape (3,), but got {q.shape}")
    if not vector.shape == (3,):
        raise ValueError(f"`vector` must have shape (3,), but got {vector.shape}")

    if isinstance(field, MagneticField_Cylindrical):
        v_XYZ, q_XYZ = find_vector_and_q_cyl_to_cart(vector, q)
    else:
        v_XYZ, q_XYZ = vector, q

    unitv_XYZ = v_XYZ / np.linalg.norm(v_XYZ)

    q_XYZ_minus = q_XYZ - 0.01 * unitv_XYZ
    q_XYZ_plus = q_XYZ + 0.01 * unitv_XYZ

    polflux_at_q_minus = field.polflux_cartesian(*q_XYZ_minus)
    polflux_at_q = field.polflux_cartesian(*q_XYZ)
    polflux_at_q_plus = field.polflux_cartesian(*q_XYZ_plus)

    if polflux_at_q_minus < polflux_at_q < polflux_at_q_plus:
        raise ValueError(f"K_plasma is pointing out of the plasma!")
    elif polflux_at_q_plus < polflux_at_q < polflux_at_q_minus:
        pass
    else:
        log.warning(
            f"Warning: Unable to check if K_plasma is pointing in or out of the plasma!"
        )


##################################################
#
# BOUNDARY CONDITION CODES
#
##################################################


def find_H_bar(
    mode_flag: Literal[1, -1],
    K_parallel: ArrayLike,
    K_binormal: ArrayLike,
    K_normal: ArrayLike,
    parallel_unitvector: FloatArray,
    binormal_unitvector: FloatArray,
    normal_unitvector: FloatArray,
    B_X: ArrayLike,
    B_Y: ArrayLike,
    B_Z: ArrayLike,
    launch_angular_frequency: float,
    electron_density: ArrayLike,
    temperature: Optional[ArrayLike] = None,
) -> ArrayLike:

    K_0 = angular_frequency_to_wavenumber(launch_angular_frequency)
    K_cartesian = (
        K_parallel * parallel_unitvector
        + K_binormal * binormal_unitvector
        + K_normal * normal_unitvector
    )
    K_X, K_Y, K_Z = K_cartesian
    K_magnitude = np.sqrt(K_X**2 + K_Y**2 + K_Z**2)
    B_magnitude = np.sqrt(B_X**2 + B_Y**2 + B_Z**2)
    sin_theta_m = (K_X * B_X + K_Y * B_Y + K_Z * B_Z) / (K_magnitude * B_magnitude)
    sin_theta_m_sq = sin_theta_m**2

    Booker_alpha, Booker_beta, Booker_gamma = find_Booker_terms(
        launch_angular_freq=launch_angular_frequency,
        B_total=B_magnitude,
        sin_theta_m_sq=sin_theta_m_sq,
        electron_density=electron_density,
        temperature=temperature,
    )

    H_discriminant = find_Booker_discriminant(
        launch_angular_freq=launch_angular_frequency,
        B_total=B_magnitude,
        sin_theta_m_sq=sin_theta_m_sq,
        electron_density=electron_density,
        temperature=temperature,
    )

    return K_magnitude**2 + K_0**2 * (
        (Booker_beta - mode_flag * np.sqrt(H_discriminant)) / (2 * Booker_alpha)
    )


def find_K_plasma(
    boundary_flag: VALID_BOUNDARY_FLAGS,
    q_entry: FloatArray,
    K_vacuum: FloatArray,
    field: VALID_FIELDS,
    hamiltonian: Hamiltonian,
) -> FloatArray:
    """NOTE: if K_vacuum is in cyl (cart), K_plasma is in cyl (cart)"""

    if boundary_flag in ["continuous", None]:
        K_plasma = K_vacuum
    else:  # elif boundary_flag == "discontinuous":

        # The cylindrical calculations which follow are all special cases
        # of the cartesian calculations (when Y=0). Hence, for example,
        # we have dp/dX = dp/dR and dp/dY = dp/dzeta
        if isinstance(field, MagneticField_Cylindrical):
            cart = False
            delta_R, delta_Z = hamiltonian.spacings["q0"], hamiltonian.spacings["q2"]
            dp_dX = field.d_polflux_dR(*q_entry, delta_R=delta_R)
            dp_dY = np.zeros_like(dp_dX)
            dp_dZ = field.d_polflux_dZ(*q_entry, delta_Z=delta_Z)
        else:
            cart = True
            delta_X, delta_Y, delta_Z = (
                hamiltonian.spacings["q0"],
                hamiltonian.spacings["q1"],
                hamiltonian.spacings["q2"],
            )
            dp_dX = field.d_polflux_dX(*q_entry, delta_X=delta_X)
            dp_dY = field.d_polflux_dY(*q_entry, delta_Y=delta_Y)
            dp_dZ = field.d_polflux_dZ(*q_entry, delta_Z=delta_Z)

        polflux_at_boundary = field.polflux(*q_entry)
        mode_flag = hamiltonian.mode_flag
        angular_freq = hamiltonian.angular_frequency
        K0 = hamiltonian.wavenumber
        electron_density_p = hamiltonian.density(polflux_at_boundary)
        temperature = (
            hamiltonian.temperature(polflux_at_boundary)
            if hamiltonian.temperature
            else None
        )

        # Getting magnetic field quantities
        B_vec, b_hat, B_magnitude = field.all(*q_entry)

        # Check plasma cutoff and hybrid frequencies
        (
            normalised_plasma_freq,
            normalised_gyro_freq,
            normalised_LH_cutoff_freq,
            normalised_RH_cutoff_freq,
            normalised_UH_freq,
        ) = find_normalised_angular_freqs(
            angular_freq, B_magnitude, electron_density_p, temperature
        )

        log.debug(f"""
        Finding K at the plasma entry point with {boundary_flag} boundary conditions
        ##################################################
        #
        # Calculated at {"[X, Y, Z]" if cart else "[R, zeta, Z]"} = {q_entry}:
        #   - B_vec = {B_vec}
        #   - b_hat = {b_hat}
        #   - |B| = {B_magnitude}
        #
        #   - w_pe / w_launch = {normalised_plasma_freq}
        #   - w_ce / w_launch = {normalised_gyro_freq}
        #   - w_L  / w_launch = {normalised_LH_cutoff_freq}
        #   - w_R  / w_launch = {normalised_RH_cutoff_freq}
        #   - w_UH / w_launch = {normalised_UH_freq}
        #
        #   - {"d(polflux)/dX" if cart else "d(polflux)/dR"} = {dp_dX}
        #   - {"d(polflux)/dY" if cart else "d(polflux)/dzeta"} = {dp_dY}
        #   - {"d(polflux)/dZ" if cart else "d(polflux)/dZ"} = {dp_dZ}
        #
        ##################################################
        """)

        # TODO -- need to rewrite andf refactor this properly. 12 Nov
        mode_flag_sign = 1  # find_mode_flag_sign(electron_density_p, B_magnitude, launch_angular_frequency, temperature)
        if (
            (mode_flag_sign * mode_flag == 1 and normalised_plasma_freq >= 1)
            or (mode_flag_sign * mode_flag == -1 and normalised_LH_cutoff_freq >= 1)
            or (
                mode_flag_sign * mode_flag == -1
                and normalised_RH_cutoff_freq >= 1
                and normalised_UH_freq <= 1
            )
        ):
            raise ValueError(
                "Error: cut-off freq higher than beam freq on plasma side of plasma-vac boundary"
            )

        if mode_flag_sign * mode_flag == 1 and normalised_plasma_freq >= 1:
            raise ValueError(
                f"Cut-off freq higher than beam freq on plasma side of plasma-vacuum boundary: w_plasma / w_launch = {normalised_plasma_freq} >= 1"
            )
        elif mode_flag_sign * mode_flag == -1 and normalised_LH_cutoff_freq >= 1:
            raise ValueError(
                f"Cut-off freq higher than beam freq on plasma side of plasma-vacuum boundary: w_L / w_launch = {normalised_LH_cutoff_freq} >= 1"
            )
        elif (
            mode_flag_sign * mode_flag == -1
            and normalised_RH_cutoff_freq >= 1
            and normalised_UH_freq <= 1
        ):
            raise ValueError(
                f"Cut-off freq higher than beam freq on plasma side of plasma-vacuum boundary: w_R / w_launch = {normalised_RH_cutoff_freq} >= 1 and w_UH / w_launch = {normalised_UH_freq} <= 1"
            )

        # In our derivations, we find three vectors which are parallel to
        # the flux surface by considering three displacements in X, Y, and Z
        # to be zero, respectively. Selecting two of these vectors yields a
        # linearly independent (but not necessarily orthogonal) basis which
        # locally parametrises the surface. Thus, what we do is to calculate
        # the vector normal to these two vectors, and then calculate the
        # binormal vector (by using one of the original vectors and the
        # normal vector). This allows us to calculate K_plasma from K_vacuum
        # by projecting K_vacuum onto the two parallel vectors. The last
        # component (normal to the surface) can then be found by solving
        # the dispersion relation H = 0

        # parallel_unitvector1 corresponds to delta_X = 0 and is equal to
        # np.array([-dp_dY, dp_dX, 0]) / np.sqrt( dp_dX**2 + dp_dY**2 )
        # this is essentially equivalent to the toroidal unit vector
        # parallel_unitvector2 corresponds to delta_Z = 0 and is equal to
        # np.array([-dp_dZ, 0, dp_dX]) / np.sqrt( dp_dX**2 + dp_dZ**2 )
        # this is essentially equivalent to the poloidal unit vector
        # normal_unitvector = parallel_vector1 x parallel_vector2
        # binormal_unitvector = parallel_vector1 x normal_vector
        # this must reduce to parallel_unitvector2/the poloidal unit
        # vector when dH/dY = dH/dzeta = 0
        parallel_unitvector1 = np.array([-dp_dY, dp_dX, 0]) / np.sqrt(
            dp_dX**2 + dp_dY**2
        )
        parallel_unitvector2 = np.array([-dp_dZ, 0, dp_dX]) / np.sqrt(
            dp_dX**2 + dp_dZ**2
        )
        normal_vector = np.cross(parallel_unitvector1, parallel_unitvector2)
        normal_unitvector = normal_vector / np.linalg.norm(normal_vector)
        binormal_vector = np.cross(parallel_unitvector1, normal_unitvector)
        binormal_unitvector = binormal_vector / np.linalg.norm(binormal_vector)

        # Checking to see if the normal vector points into or out of the plasma
        # and flipping the sign (if necessary) to ensure it always points
        # inward
        q_check = q_entry + 0.01 * normal_unitvector
        polflux_check = field.polflux(*q_check)
        if polflux_at_boundary < polflux_check:
            normal_unitvector = -normal_unitvector

        # Now we convert `K_v` properly (because in our convention, `K_zeta`
        # is dimensionless) by dividing it by `q_R`. We call this K_vacuum_actual
        # This is necessary to calculate the mismatch angle `theta_m`, and is itself
        # necessary to calculate the Booker terms
        if isinstance(field, MagneticField_Cylindrical):
            K_vacuum_actual = K_vacuum / np.array(
                [1, q_entry[0], 1]
            )  # because `K_zeta` is dimensionless, so we convert it to `K_T`
            sin_theta_m = np.dot(b_hat, K_vacuum_actual) / np.linalg.norm(
                K_vacuum_actual
            )
        else:
            K_vacuum_actual = K_vacuum
            sin_theta_m = np.dot(b_hat, K_vacuum_actual) / np.linalg.norm(
                K_vacuum_actual
            )

        sin_theta_m_sq = sin_theta_m**2

        Booker_alpha, Booker_beta, Booker_gamma = find_Booker_terms(
            launch_angular_freq=angular_freq,
            B_total=B_magnitude,
            sin_theta_m_sq=sin_theta_m_sq,
            electron_density=electron_density_p,
            temperature=temperature,
        )

        # Now we decompose `K_v_actual` into two (orthogonal) components along
        # the flux surface, and one component normal to the flux surface. This is
        # done using the dispersion relation H = 0. We first guess what it could be,
        # and then use that to numerically compute what it actually is. This guess will
        # be exact if theta_m = 0
        K_parallel_actual = np.dot(K_vacuum_actual, parallel_unitvector1)  # toroidal
        K_binormal_actual = np.dot(K_vacuum_actual, binormal_unitvector)  # poloidal

        H_discriminant = find_Booker_discriminant(
            launch_angular_freq=angular_freq,
            B_total=B_magnitude,
            sin_theta_m_sq=sin_theta_m_sq,
            electron_density=electron_density_p,
            temperature=temperature,
        )
        K_normal_plasma_actual_initial_guess = np.sqrt(
            abs(
                K_parallel_actual**2
                + K_binormal_actual**2
                + K0**2
                * (
                    (Booker_beta - mode_flag * np.sqrt(H_discriminant))
                    / (2 * Booker_alpha)
                )
            )
        )

        def find_H_bar_wrapper(K_normal_plasma_guess: ArrayLike):
            return find_H_bar(
                mode_flag=mode_flag,  # type: ignore
                K_parallel=K_parallel_actual,
                K_binormal=K_binormal_actual,
                K_normal=K_normal_plasma_guess,
                parallel_unitvector=parallel_unitvector1,
                binormal_unitvector=binormal_unitvector,
                normal_unitvector=normal_unitvector,
                B_X=B_vec[0],
                B_Y=B_vec[1],
                B_Z=B_vec[2],
                launch_angular_frequency=angular_freq,
                electron_density=electron_density_p,
                temperature=temperature,
            )

        # Comments from the original function code:
        #   This will fail if the beam is too glancing such that there
        #   is no possible `K_normal_plasma` that satisfies H = 0
        K_normal_plasma_actual = newton(
            find_H_bar_wrapper,
            K_normal_plasma_actual_initial_guess,
            tol=1e-10,
            maxiter=5000,
        )

        # After finding `K_normal_plasma`, we find `K_plasma`
        K_plasma_actual = (
            K_parallel_actual * parallel_unitvector1
            + K_binormal_actual * binormal_unitvector
            + K_normal_plasma_actual * normal_unitvector
        )

        # To make sure K_plasma is valid, we
        #   i) check if it points into the plasma; and
        #   ii) check if it satisfies H = 0
        check_vector_pointing_into_plasma(q_entry, K_plasma_actual, field)

        H_bar_check = find_H_bar(
            mode_flag=mode_flag,  # type: ignore
            K_parallel=K_plasma_actual[0],
            K_binormal=K_plasma_actual[1],
            K_normal=K_plasma_actual[2],
            parallel_unitvector=np.array([1, 0, 0]),
            binormal_unitvector=np.array([0, 1, 0]),
            normal_unitvector=np.array([0, 0, 1]),
            B_X=B_vec[0],
            B_Y=B_vec[1],
            B_Z=B_vec[2],
            launch_angular_frequency=angular_freq,
            electron_density=electron_density_p,
            temperature=temperature,
        )

        if abs(H_bar_check) > 1e-3:
            raise RuntimeError(
                f"Unable to find `K_plasma` with discontinuous boundary conditions: \nH_bar_check = {H_bar_check}"
            )

        # Remember that for cylindrical calculations, `K_zeta` is the
        # dimensionless mode number
        if isinstance(field, MagneticField_Cylindrical):
            K_plasma = K_plasma_actual * np.array([1, q_entry[0], 1])
        else:
            K_plasma = K_plasma_actual

    return K_plasma


def find_Psi_3D_plasma(
    boundary_flag: VALID_BOUNDARY_FLAGS,
    q_entry: FloatArray,
    K_vacuum: FloatArray,
    K_plasma: FloatArray,
    Psi_3D_vacuum_labframe: ComplexFloatArray,
    field: VALID_FIELDS,
    hamiltonian: Hamiltonian,
) -> ComplexFloatArray:

    cart = False if isinstance(field, MagneticField_Cylindrical) else True

    # No boundary conditions to be applied
    if boundary_flag is None:
        log.debug(f"""
        Finding Psi at the plasma entry point with {boundary_flag} boundary conditions
        ##################################################
        #""")

        Psi_XX_p = Psi_3D_vacuum_labframe[0, 0]
        Psi_XY_p = Psi_3D_vacuum_labframe[0, 1]
        Psi_XZ_p = Psi_3D_vacuum_labframe[0, 2]
        Psi_YY_p = Psi_3D_vacuum_labframe[1, 1]
        Psi_YZ_p = Psi_3D_vacuum_labframe[1, 2]
        Psi_ZZ_p = Psi_3D_vacuum_labframe[2, 2]

    # Continuous or discontinuous boundary conditions to be applied
    else:
        # The cylindrical calculations which follow are all special cases
        # of the cartesian calculations (when Y=0). Hence, for example,
        # we have dp/dX = dp/dR and dp/dY = dp/dzeta
        if not cart:
            dH = hamiltonian.derivatives(q_entry, K_plasma)

            delta_R, delta_Z = hamiltonian.spacings["q0"], hamiltonian.spacings["q2"]
            derivatives = {
                # First derivatives of the Hamiltonian
                "dH_dR": dH["dH_dR"],
                "dH_dzeta": np.zeros_like(dH["dH_dR"]),
                "dH_dZ": dH["dH_dZ"],
                "dH_dKR": dH["dH_dKR"],
                "dH_dKzeta": dH["dH_dKzeta"],
                "dH_dKZ": dH["dH_dKZ"],
                # First derivatives of poloidal flux
                "dp_dR": (tmp := field.d_polflux_dR(*q_entry, delta_R=delta_R)),  # type: ignore
                "dp_dzeta": np.zeros_like(tmp),
                "dp_dZ": field.d_polflux_dZ(*q_entry, delta_Z=delta_Z),  # type: ignore
                # Second derivatives of poloidal flux
                "d2p_dR2": field.d2_polflux_dR2(*q_entry, delta_R=delta_R),  # type: ignore
                "d2p_dzeta2": dH["dH_dzeta"],
                "d2p_dZ2": field.d2_polflux_dZ2(*q_entry, delta_Z=delta_Z),  # type: ignore
                "d2p_dRdzeta": dH["dH_dzeta"],
                "d2p_dRdZ": field.d2_polflux_dRdZ(*q_entry, delta_R=delta_R, delta_Z=delta_Z),  # type: ignore
                "d2p_dzetadZ": dH["dH_dzeta"],
            }

        else:
            derivatives = hamiltonian.derivatives(q_entry, K_plasma)

            delta_X, delta_Y, delta_Z = (
                hamiltonian.spacings["q0"],
                hamiltonian.spacings["q1"],
                hamiltonian.spacings["q2"],
            )
            derivatives.update(
                {
                    # First derivatives of poloidal flux
                    "dp_dX": field.d_polflux_dX(*q_entry, delta_X=delta_X),  # type: ignore
                    "dp_dY": field.d_polflux_dY(*q_entry, delta_Y=delta_Y),  # type: ignore
                    "dp_dZ": field.d_polflux_dZ(*q_entry, delta_Z=delta_Z),  # type: ignore
                    # Second derivatives of poloidal flux
                    "d2p_dX2": field.d2_polflux_dX2(*q_entry, delta_X=delta_X),  # type: ignore
                    "d2p_dY2": field.d2_polflux_dY2(*q_entry, delta_Y=delta_Y),  # type: ignore
                    "d2p_dZ2": field.d2_polflux_dZ2(*q_entry, delta_Z=delta_Z),  # type: ignore
                    "d2p_dXdY": field.d2_polflux_dXdY(*q_entry, delta_X=delta_X, delta_Y=delta_Y),  # type: ignore
                    "d2p_dXdZ": field.d2_polflux_dXdZ(*q_entry, delta_X=delta_X, delta_Z=delta_Z),  # type: ignore
                    "d2p_dYdZ": field.d2_polflux_dYdZ(*q_entry, delta_Y=delta_Y, delta_Z=delta_Z),  # type: ignore
                }
            )

        _printmsg = "\n".join(
            f"        #   - {k} = {v}" for k, v in derivatives.items()
        )
        log.debug(f"""
        Finding Psi at the plasma entry point with {boundary_flag} boundary conditions
        ##################################################
        # Derivatives at {"[X, Y, Z]" if cart else "[R, zeta, Z]"} = {q_entry} with {"[K_X, K_Y, K_Z]" if cart else "[K_R, K_zeta, K_Z]"} = {K_plasma}: \n{_printmsg}
        #""")

        # Gradients at the plasma-vacuum boundary can be finnicky, so best
        # to double check
        for k, v in derivatives.items():
            if np.isnan(v):
                raise ValueError(f"{k} is {v}")

        # Converting the derivatives into variables for readability (later)
        dH_dX = derivatives["dH_dX" if cart else "dH_dR"]
        dH_dY = derivatives["dH_dY" if cart else "dH_dzeta"]
        dH_dZ = derivatives["dH_dZ"]
        dH_dKx = derivatives["dH_dKX" if cart else "dH_dKR"]
        dH_dKy = derivatives["dH_dKY" if cart else "dH_dKzeta"]
        dH_dKz = derivatives["dH_dKZ"]

        dp_dX = derivatives["dp_dX" if cart else "dp_dR"]
        dp_dY = derivatives["dp_dY" if cart else "dp_dzeta"]
        dp_dZ = derivatives["dp_dZ"]

        d2p_dX2 = derivatives["d2p_dX2" if cart else "d2p_dR2"]
        d2p_dY2 = derivatives["d2p_dY2" if cart else "d2p_dzeta2"]
        d2p_dZ2 = derivatives["d2p_dZ2" if cart else "d2p_dZ2"]
        d2p_dXdY = derivatives["d2p_dXdY" if cart else "d2p_dRdzeta"]
        d2p_dXdZ = derivatives["d2p_dXdZ" if cart else "d2p_dRdZ"]
        d2p_dYdZ = derivatives["d2p_dYdZ" if cart else "d2p_dzetadZ"]

        # At the plasma-vacuum boundary, we have two Psi matrices:
        # one corresponding to Psi in the vacuum (entry), and the
        # other corresponding to Psi in the plasma (initial). We
        # denote these by the subscripts 'v' and 'p' respectively
        Psi_XX_v = Psi_3D_vacuum_labframe[0, 0]
        Psi_XY_v = Psi_3D_vacuum_labframe[0, 1]
        Psi_XZ_v = Psi_3D_vacuum_labframe[0, 2]
        Psi_YY_v = Psi_3D_vacuum_labframe[1, 1]
        Psi_YZ_v = Psi_3D_vacuum_labframe[1, 2]
        Psi_ZZ_v = Psi_3D_vacuum_labframe[2, 2]

        # Now we set up the interface matrix using 6 linearly
        # independent equations to obtain a relation between the
        # entries of "Psi_v" and "Psi_p". Note that this matrix
        # differs slightly, even after applying toroidal symmetry
        # arguments, from the original interface matrix for
        # cylindrical Scotty. However, these differences, in
        # essence, disappear when one applies row operations
        # (like partial Gaussian elimination)
        interface_matrix = np.array(
            [
                [dp_dY**2, -2 * dp_dX * dp_dY, 0, dp_dX**2, 0, 0],
                [dp_dZ**2, 0, -2 * dp_dX * dp_dZ, 0, 0, dp_dX**2],
                [
                    dp_dZ**2,
                    2 * dp_dZ**2,
                    -2 * dp_dZ * (dp_dX + dp_dY),
                    dp_dZ**2,
                    -2 * dp_dZ * (dp_dX + dp_dY),
                    (dp_dX + dp_dY) ** 2,
                ],
                [dH_dKx, dH_dKy, dH_dKz, 0, 0, 0],
                [0, dH_dKx, 0, dH_dKy, dH_dKz, 0],
                [0, 0, dH_dKx, 0, dH_dKy, dH_dKz],
            ],
            dtype=np.float64,
        )

        log.debug(f"""
        #
        #   - interface matrix =
        #        {arr2str(interface_matrix[0])}
        #        {arr2str(interface_matrix[1])}
        #        {arr2str(interface_matrix[2])}
        #        {arr2str(interface_matrix[3])}
        #        {arr2str(interface_matrix[4])}
        #        {arr2str(interface_matrix[5])}
        #
        #   - np.linalg.cond(interface matrix) = {np.linalg.cond(interface_matrix)}
        #""")

        # For discontinuous boundary conditions, we have that
        # K_vacuum =/= K_plasma in general, and this introduces
        # an `eta` term for each case delta_X, delta_Y, delta_Z = 0
        # corresponding to displacements in the YZ, XZ, and XY-planes
        # respectively, which also corresponds to `eta_YZ` (not used),
        # `eta_XZ`, and `eta_XY` respectively. The `eta_YZ` term is
        # unused because the decomposing the interface matrix in this
        # way leads to a singular matrix (because the resultant columns
        # somehow become linearly dependent). To circumvent this, we
        # elect to use `eta_XYZ` instead, which represents a general
        # displacement in any direction

        # On the other hand, for continuous boundary conditions,
        # K_vacuum = K_plasma and so this simplifies the calculation
        # quite a bit since no `eta`s are necessary since the `K`
        # terms cancel out nicely. We set these to `0` just to
        # (double) enforce the fact that the terms should cancel out

        delta_K = K_vacuum - K_plasma
        if boundary_flag == "continuous":
            eta_XY = eta_XZ = eta_XYZ = 0
        else:  # boundary_flag == "discontinuous"
            eta_XY = (
                -0.5
                * (
                    d2p_dX2 * dp_dY**2
                    + 2 * d2p_dXdY * dp_dX * dp_dY
                    + d2p_dY2 * dp_dX**2
                )
                / (dp_dX**2 + dp_dY**2)
            )
            eta_XZ = (
                -0.5
                * (
                    d2p_dX2 * dp_dZ**2
                    + 2 * d2p_dXdZ * dp_dX * dp_dZ
                    + d2p_dZ2 * dp_dX**2
                )
                / (dp_dX**2 + dp_dZ**2)
            )
            eta_XYZ = (
                -0.5
                * (
                    d2p_dX2 * dp_dZ**2
                    + d2p_dY2 * dp_dZ**2
                    + d2p_dZ2 * (dp_dX + dp_dY) ** 2
                    + 2 * d2p_dXdY * dp_dZ**2
                    + 2 * d2p_dXdZ * dp_dZ * (dp_dX + dp_dY)
                    + 2 * d2p_dYdZ * dp_dZ * (dp_dX + dp_dY)
                )
                / (dp_dX**2 + dp_dY**2 + dp_dZ**2)
            )

            log.debug(f"""
        #
        #   - eta_XY  = {eta_XY}
        #   - eta_XZ  = {eta_XZ}
        #   - eta_XYZ = {eta_XYZ}
        #""")

        Psi_v_vector = np.array(
            [Psi_XX_v, Psi_XY_v, Psi_XZ_v, Psi_YY_v, Psi_YZ_v, Psi_ZZ_v], dtype=complex
        )
        RHS_vector = np.array(
            [
                np.dot(Psi_v_vector, interface_matrix[0])
                + 2 * eta_XY * np.dot(delta_K, np.array([dp_dX, dp_dY, 0])),
                np.dot(Psi_v_vector, interface_matrix[1])
                + 2 * eta_XZ * np.dot(delta_K, np.array([dp_dX, 0, dp_dZ])),
                np.dot(Psi_v_vector, interface_matrix[2])
                + 2 * eta_XYZ * np.dot(delta_K, np.array([dp_dX, dp_dY, dp_dZ])),
                -dH_dX,
                -dH_dY,
                -dH_dZ,
            ],
            dtype=complex,
        )

        # We access the first 3 items twice because they are tuples
        log.debug(f"""
        #
        #   - RHS vector =
        #        {np.real(RHS_vector[0])} + {np.imag(RHS_vector[0])}j
        #        {np.real(RHS_vector[1])} + {np.imag(RHS_vector[1])}j
        #        {np.real(RHS_vector[2])} + {np.imag(RHS_vector[2])}j
        #        {RHS_vector[3]}
        #        {RHS_vector[4]}
        #        {RHS_vector[5]}
        #""")

        # The interface matrix is another way to say that we're solving
        # six linear equations relating the components of `Psi` in vacuum
        # and in plasma. What this looks like is thus:
        #   interface_matrix * Psi_p_components = RHS_vector
        #
        # There are (numerically) two ways to solve this:
        #   i)  invert `interface_matrix` and left-multiply throughout; or
        #   ii) perform Gaussian elimination (i.e. RREF)
        #
        # Previously, we used to do i). However, it turns out that ii) is
        # a much better way of solving i) because of several reasons, chief
        # of which being that numerical inaccuracies tend to creep in when the
        # norm of the rows of the matrix differ by several orders of magnitude,
        # which is the case most of the time:
        #   || rows with poloidal flux derivatives || >> || rows with H derivatives ||
        #
        # Furthermore, it is actually more computationally expensive to
        # calculate the inverse than to solve the system via Gaussian,
        # elimination though this difference is negligible in our case
        # due to the (relatively) small size of the matrix (6x6)

        [Psi_XX_p, Psi_XY_p, Psi_XZ_p, Psi_YY_p, Psi_YZ_p, Psi_ZZ_p] = np.linalg.solve(
            interface_matrix, RHS_vector
        )

    # Reconstructing `Psi` matrix
    Psi_3D_plasma_labframe = np.array(
        [
            [Psi_XX_p, Psi_XY_p, Psi_XZ_p],
            [Psi_XY_p, Psi_YY_p, Psi_YZ_p],
            [Psi_XZ_p, Psi_YZ_p, Psi_ZZ_p],
        ],
        dtype=type(Psi_XX_p),
    )

    log.debug(f"""
        #
        #   - Psi_3D_plasma_labframe
        #        {"Psi_XX_p =" if cart else "Psi_RR_p       ="} {np.real(Psi_XX_p)} + {np.imag(Psi_XX_p)}j
        #        {"Psi_XY_p =" if cart else "Psi_Rzeta_p    ="} {np.real(Psi_XY_p)} + {np.imag(Psi_XY_p)}j
        #        {"Psi_XZ_p =" if cart else "Psi_RZ_p       ="} {np.real(Psi_XZ_p)} + {np.imag(Psi_XZ_p)}j
        #        {"Psi_YY_p =" if cart else "Psi_zetazeta_p ="} {np.real(Psi_YY_p)} + {np.imag(Psi_YY_p)}j
        #        {"Psi_YZ_p =" if cart else "Psi_zetaZ_p    ="} {np.real(Psi_YZ_p)} + {np.imag(Psi_YZ_p)}j
        #        {"Psi_ZZ_p =" if cart else "Psi_ZZ_p       ="} {np.real(Psi_ZZ_p)} + {np.imag(Psi_ZZ_p)}j
        #
        ##################################################
        """)  # type: ignore  # pyright: ignore[reportPossiblyUnbound]

    return Psi_3D_plasma_labframe


def apply_boundary_conditions(
    ray_tracing_flag: bool,
    boundary_flag: VALID_BOUNDARY_FLAGS,
    q_vacuum_entry_cartesian: FloatArray,
    K_vacuum_entry_cartesian: FloatArray,
    Psi_3D_vacuum_entry_labframe: Optional[ComplexFloatArray],
    field: VALID_FIELDS,
    hamiltonian: Hamiltonian,
) -> Tuple[FloatArray, Optional[ComplexFloatArray]]:
    r"""Apply boundary conditions at the plasma-vacuum boundary where
    the electron density profile can be: (i) continuous and differentiable
    (None); (ii) continuous but not differentiable ("continuous"); or
    (iii) discontinuous and not differentiable ("discontinuous).

    For (i), no boundary conditions are applied, and the
    `q`, `K`, `Psi_3D` parameters are directly returned (and later fed
    into the solver). For (ii), the `continuous` boundary condition is applied
    only for `Psi_3D_vacuum` and `Psi_3D_plasma`, while we have that
    `K_vacuum` == `K_plasma`. For (iii), `discontinuous` boundary condition is
    applied to find both `K_plasma` and `Psi_3D_plasma`.

    Returns `K_plasma` and `Psi_3D_plasma` in cylindrical (cartesian) if
    the field type is cylindrical (cartesian).
    """

    log.debug(f"Applying boundary conditions")

    # Note that all calculations, unlike in `launch.py`, are done
    # in their own coordinate systems (for abstraction purposes).
    # Specifically, we assume that the RZ-plane corresponds to the
    # XZ-plane, i.e. Y=0, so that cylindrical components/derivatives
    # correspond exactly to their cartesian counterparts. In essence,
    # this means that the calculations performed in cylindrical and
    # cartesian coordinates are the same, but one must be careful because
    # parts of the following calculation require calling functions or
    # calculating quantities that lack toroidal symmetry, for instance:
    # `B_X`, `B_Y` vs. `B_R`, `B_T`

    # Getting important quantities
    if isinstance(field, MagneticField_Cylindrical):
        cart = False
        q_entry = find_q_labframe_cart_to_cyl(q_vacuum_entry_cartesian)
        K_vacuum = find_K_labframe_cart_to_cyl(
            K_vacuum_entry_cartesian, q_vacuum_entry_cartesian
        )
    else:
        cart = True
        q_entry = q_vacuum_entry_cartesian
        K_vacuum = K_vacuum_entry_cartesian

    # Find `K_plasma`
    K_plasma = find_K_plasma(
        boundary_flag=boundary_flag,
        q_entry=q_entry,
        K_vacuum=K_vacuum,
        field=field,
        hamiltonian=hamiltonian,
    )

    # Find `Psi_3D_labframe`
    if ray_tracing_flag:
        Psi_3D_plasma = None
    else:
        Psi_3D_plasma = find_Psi_3D_plasma(
            boundary_flag=boundary_flag,
            q_entry=q_entry,
            K_vacuum=K_vacuum,
            K_plasma=K_plasma,
            Psi_3D_vacuum_labframe=cast(
                ComplexFloatArray, Psi_3D_vacuum_entry_labframe
            ),
            field=field,
            hamiltonian=hamiltonian,
        )

    return K_plasma, Psi_3D_plasma
