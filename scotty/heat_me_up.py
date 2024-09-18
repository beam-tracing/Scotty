# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 16:49:19 2024

@author: jz271
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import datatree
import scipy
from scipy.interpolate import UnivariateSpline
from scotty.analysis import beam_width
from scotty.geometry import _make_rect_spline
from scipy.integrate import cumulative_trapezoid  # for integration of arrays
from plasmapy.dispersion import plasma_dispersion_func
from scotty.fun_general import dot


def hot_dielectric(dt, Te_along_beam, m_range, omega):
    # input: Te_along_beam: temperature along the beam in keV; m_range: the amount of orders checked (usually 5 is enough);
    # omega: beam angular frequency

    # Reference: lecVII hot plasma
    # assume only electrons species

    # setting up parameters
    tau = dt.analysis.tau.data
    B_magnitude = dt.analysis.B_magnitude.data
    K_magnitude = dt.analysis.K_magnitude.data
    poloidal_flux = dt.analysis.poloidal_flux.data
    theta_m = dt.analysis.theta_m.data
    ne = dt.analysis.electron_density.data * 10 ** 19
    m_e = 9.11e-31
    q_e = 1.6021e-19
    c = 2.998e8
    epsilon_0 = 8.854e-12
    Te_along_beam = Te_along_beam * 1000 * q_e
    K_perp = K_magnitude * np.sin(theta_m)
    K_para = K_magnitude * np.cos(theta_m)

    ### non-relativistic data
    # omega_ce = q_e * B_magnitude / m_e
    # omega_pe = np.sqrt(ne * q_e ** 2 / m_e / epsilon_0)
    # b_s = K_perp ** 2 * Te_along_beam / m_e / omega_ce ** 2
    # v_te = np.sqrt(2 * Te_along_beam / m_e)

    ### consider relativistic effects
    v_te = np.sqrt(2 * Te_along_beam / m_e - 0.5 * (2 * Te_along_beam / m_e)
                   ** 2 / c ** 2)  # relativistic correction considered
    lorentz = 1 / np.sqrt(1 - v_te ** 2 / c ** 2)
    omega_ce = q_e * B_magnitude / m_e / lorentz
    omega_pe = np.sqrt(ne * q_e ** 2 / m_e / epsilon_0 / lorentz)
    b_s = K_perp ** 2 * Te_along_beam / m_e / lorentz / omega_ce ** 2

    L = len(theta_m)  # length of data points

    def zeta(m):  # wave frequency after doppler shift due to electron thermal speed, scaled by Vte
        return (omega - m * omega_ce) / (np.abs(K_para) * v_te)

    # modified Bessel function setup
    def real_integrand(x, b_s, m):
        return np.real(np.exp(b_s * np.cos(x) - 1j * m * x)) / 2 / np.pi

    def imag_integrand(x, b_s, m):
        return np.imag(np.exp(b_s * np.cos(x) - 1j * m * x)) / 2 / np.pi

    def m_bessel(m, b_s):
        real = scipy.integrate.quad(
            real_integrand, -np.pi, np.pi, args=(b_s, m))[0]
        imag = scipy.integrate.quad(
            imag_integrand, -np.pi, np.pi, args=(b_s, m))[0]
        return real + 1j * imag

    # computation of dielectric tensor
    epsi = np.zeros((2 * m_range + 1, 3, 3, L), dtype=np.complex_)
    C1 = omega_pe ** 2 / (omega * np.abs(K_para) * v_te)
    C2 = omega_pe ** 2 / (omega * K_para * v_te)
    for m in range(- m_range, m_range + 1):
        if m == m_range:  # 1 belongs to the 0th order
            epsi[m][0][0] += 1
            epsi[m][1][1] += 1
            epsi[m][2][2] += 1
        I = []
        dI_db = []
        for i in range(len(b_s)):
            I.append(m_bessel(m, b_s[i]))
            dI_db.append((m_bessel(m-1, b_s[i]) + m_bessel(m+1, b_s[i])) / 2)

            # check for resonance
            if i != 0 and ((zeta(m)[i]) - 1) * ((zeta(m)[i-1] - 1)) < 0:
                print(str(m) + "th order detected at index " +
                      str(i) + ", tau position " + str(tau[i]) + "and poloidal flux " + str(poloidal_flux[i]))

        I, dI_db = np.array(I), np.array(dI_db)
        epsi[m][0][0] += C1 * m ** 2 * I / b_s * \
            np.exp(-b_s) * plasma_dispersion_func(zeta(m))
        epsi[m][0][1] += 1j * C1 * m * \
            (dI_db - I) * np.exp(-b_s) * plasma_dispersion_func(zeta(m))
        epsi[m][0][2] += -C2 * np.sqrt(2 / b_s) * m * I * np.exp(-b_s) * (
            1 + zeta(m) * plasma_dispersion_func(zeta(m)))
        epsi[m][1][1] += C1 * (m ** 2 * I / b_s + 2 * b_s * (I - dI_db)) * \
            np.exp(-b_s) * plasma_dispersion_func(zeta(m))
        epsi[m][1][2] += 1j * C2 * np.sqrt(2 * b_s) * (dI_db - I) * np.exp(-b_s) * (
            1 + zeta(m) * plasma_dispersion_func(zeta(m)))
        epsi[m][2][2] += 2 * C1 * I * \
            np.exp(-b_s) * zeta(m) * (1 + zeta(m) *
                                      plasma_dispersion_func(zeta(m)))
        epsi[m][1][0] -= epsi[m][0][1]
        epsi[m][2][0] += epsi[m][0][2]
        epsi[m][2][1] -= epsi[m][1][2]

    Epsi = np.sum(epsi, axis=0)
    Epsi_t = np.transpose(Epsi.conj(), [1, 0, 2])  # conjugate transpose
    Epsi_h = (Epsi + Epsi_t) / 2  # hermitian component
    Epsi_a = (Epsi - Epsi_t) / 2  # antihermitian component
    # output shape: (3, 3, 102)

    # finally, output the cold plasma dielectrics for comparison
    Epsi_cold = np.zeros((3, 3, len(tau))) + 0j
    Epsi_cold[0, 0, :] = 1 - omega_pe ** 2 / (omega ** 2 - omega_ce ** 2)
    Epsi_cold[1, 1, :] = 1 - omega_pe ** 2 / (omega ** 2 - omega_ce ** 2)
    Epsi_cold[2, 2, :] = 1 - omega_pe ** 2 / omega ** 2
    Epsi_cold[0, 1, :] = - 1j * omega_pe ** 2 * \
        omega_ce / omega / (omega ** 2 - omega_ce ** 2)
    Epsi_cold[1, 0, :] = 1j * omega_pe ** 2 * \
        omega_ce / omega / (omega ** 2 - omega_ce ** 2)
        

    return Epsi_cold, Epsi_h, Epsi_a


def xy_to_u_basis_change(dt, data, data_type):
    # vector shape: 102, 3
    theta = dt.analysis.theta.data
    L = len(theta)
    M = np.zeros((3, 3, L))
    M[0, 0, :] = -np.sin(theta)
    M[0, 2, :] = np.cos(theta)
    M[1, 1, :] = 1
    M[2, 0, :] = -np.cos(theta)
    M[2, 2, :] = -np.sin(theta)
    # matrix shape: 3, 3, 102
    # initial x,y,g: -y, z, -x in Cartesian

    if data_type == "vector":
        # required vector shape: L, 3
        if np.shape(data) != (L, 3):
            raise TypeError(
                "Wrong vector shape, required vector shape (len(tau), 3)"
            )
        result = np.zeros(np.shape(data)) + 0j
        for i in range(L):
            result[i, :] = np.matmul(
                M[:, :, i], data[i, :].real) + 1j * np.matmul(M[:, :, i], data[i, :].imag)
        return result

    if data_type == "matrix":
        # required matrix shape: 3, 3, L (sorry for the difference in convention)
        if np.shape(data) != (3, 3, L):
            raise TypeError(
                "Wrong matrix shape, required vector shape (3, 3, len(tau))"
            )
        result = np.zeros(np.shape(data)) + 0j
        for i in range(L):
            result[:, :, i] = np.matmul(
                M[:, :, i], data[:, :, i].real) + 1j * np.matmul(M[:, :, i], data[:, :, i].imag)
        return result


def absorption_coef(e_hat, Epsi_a):
    # solve for gamma based on e_hat and antihermitian dielectric
    gamma = []
    for i in range(np.shape(e_hat)[0]):
        gamma.append(
            np.dot(np.matmul(Epsi_a[:, :, i], e_hat[i]), e_hat[i].conj()))
    return np.array(gamma)


def power_along_beam(dt, gamma, P0):
    tau = dt.analysis.tau.data
    dP_dtau_coeff = - 2 * gamma
    def dP_dtau(t, P):  # t is the tau index in this case, not tau itself
        i = int(np.floor(t))
        return P * dP_dtau_coeff[i] * tau[-1] / len(tau)
    P_sol = scipy.integrate.solve_ivp(dP_dtau, [0, len(
        tau) - 1], [P0], method='LSODA', t_eval=np.arange(0, len(tau), 1))
    # sometimes doesn't work, can try different methods
    return P_sol.y[0]


def poloidal_flux_periphery(dt, figure_flag=False):
    # find the poloidal_flux of peripheral rays through interpolation
    poloidal_flux_grid = dt.inputs.poloidalFlux_grid.data
    B_R = dt.inputs.R.values
    B_Z = dt.inputs.Z.values
    width = beam_width(dt.analysis.g_hat, np.array(
        [0.0, 1.0, 0.0]), dt.analysis.Psi_3D)
    beam = dt.analysis.beam
    beam_plus = beam + width
    beam_minus = beam - width

    # find the interpolated poloidal flux
    interp_order, interp_smoothing = 5, 0
    interp_poloidal_flux, psi_spline = _make_rect_spline(
        B_R, B_Z, poloidal_flux_grid, interp_order, interp_smoothing
    )

    beam_plus_flux = interp_poloidal_flux(
        beam_plus.sel(col="R").data, beam_plus.sel(col="Z").data)
    beam_minus_flux = interp_poloidal_flux(
        beam_minus.sel(col="R").data, beam_minus.sel(col="Z").data)

    if figure_flag:
        font = {'family': 'sans-serif',
                # 'weight' : 'bold',
                'size': 12}
        tau = dt.analysis.tau.data
        poloidal_flux = dt.analysis.poloidal_flux.data
        matplotlib.rc('font', **font)
        plt.figure()
        plt.plot(tau, beam_plus_flux, marker='o', label='periphery')
        plt.plot(tau, beam_minus_flux, marker='o', label='periphery')
        plt.plot(tau, poloidal_flux, marker='o', label='central')
        plt.xlabel("tau")
        plt.ylabel("Normalized poloidal flux")
        plt.legend()
    return beam_plus_flux, beam_minus_flux


def Te_along_beam_interp(Te_filename, dt):
    # find the Temperature profile along the beam path
    
    # read the file
    poloidal_flux = dt.analysis.poloidal_flux.data
    Te_data = np.fromfile(Te_filename, dtype=float, sep="   ")
    Te_data_density_array = Te_data[2::2]
    Te_data_radialcoord_array = Te_data[1::2]

    # interpolation
    Te_spl = UnivariateSpline(
        Te_data_radialcoord_array,
        Te_data_density_array)
    
    # not sure about the exact meaning of the first column, I guess it is rho (sqrt poloidal flux)
    return Te_spl(np.sqrt(poloidal_flux))


def power_deposition_Gaussian(dt, P_along_beam, gamma, beam_plus_flux, beam_minus_flux, len_rho=1000):
    poloidal_flux = dt.analysis.poloidal_flux.data
    tau = dt.analysis.tau.data

    # truncate the beam: only consider points with both upper and lower peripheral rays inside the plasma
    full_beam_inside = False
    width = []
    start_truncate, end_truncate = 0, len(poloidal_flux)
    for i in range(len(P_along_beam)):
        if beam_plus_flux[i] < 1 and beam_minus_flux[i] < 1:
            if full_beam_inside == False:
                start_truncate = i
                full_beam_inside = True
            width.append(
                np.abs(np.sqrt(beam_plus_flux[i]) - np.sqrt(beam_minus_flux[i])) / 2)
        else:
            if full_beam_inside == True:
                end_truncate = i
                full_beam_inside = False

    width = np.array(width)
    # rho of the central beam
    rho_mean = np.sqrt(poloidal_flux[start_truncate: end_truncate])

    def dV_drho_circular(rho):
        # Assume circulation poloidal cross section
        # Area = 4pi ** 2 * rho ** 2 * minor_radius * major_radius
        return ((4 * np.pi ** 2 * dt.inputs.minor_radius_a.data * rho *
                 dt.inputs.R_axis.data)).reshape((len_rho, 1))

    def Gaussian_absorption(rho):
        # ij: the absorption at ith flux surface due to the jth point on the ray
        return np.exp(-2 * np.subtract.outer(rho, rho_mean) ** 2 / width ** 2)

    rho = np.linspace(0, 1, len_rho)
    integrated_Gaussian = cumulative_trapezoid(np.transpose(
        Gaussian_absorption(rho) * dV_drho_circular(rho)), rho)

    dPi = P_along_beam[start_truncate: end_truncate] * \
        gamma[start_truncate: end_truncate] * 2 * tau[-1] / len(tau)
    Ci = dPi / integrated_Gaussian[:, -1]
    dPi_dV = np.reshape(
        np.sum(Ci * Gaussian_absorption(rho), axis=1), (len_rho, 1))
    return dPi_dV * dV_drho_circular(rho)


def power_deposition_integration(dt, P_along_beam, gamma, R_step = 0.001, theta_step = 0.1, len_rho = 1000):
    # interpolate the poloidal flux first
    poloidal_flux_grid = dt.inputs.poloidalFlux_grid.data
    B_R = dt.inputs.R.values
    B_Z = dt.inputs.Z.values
    interp_order, interp_smoothing = 5, 0
    interp_poloidal_flux, psi_spline = _make_rect_spline(
        B_R, B_Z, poloidal_flux_grid, interp_order, interp_smoothing)
    
    # data preparation
    Psi_3D = dt.analysis.Psi_3D.data
    beam = dt.analysis.beam.data
    tau = dt.analysis.tau.data
    dtau = np.delete(tau, [0]) - np.delete(tau, [-1])
    dtau = np.insert(dtau, [0], dtau[0])
    omega = dt.inputs.launch_angular_frequency.data
    g_magnitude = dt.analysis.g_magnitude_Cardano.data.real
    g_hat = dt.analysis.g_hat.data
    
    # calculation of the widths
    width_x = beam_width(dt.analysis.g_hat, np.array(
        [0.0, 1.0, 0.0]), dt.analysis.Psi_3D).data
    width_x_mag = np.sqrt(width_x[:, 0] ** 2 + width_x[:, 1] ** 2 + width_x[:, 2] ** 2)
    width_x_uvec = width_x / width_x_mag.reshape(len(tau), 1)
    width_y_uvec = np.cross(width_x, g_hat) / width_x_mag.reshape(len(tau), 1)
    width_y_mag = np.sqrt(2 / dot(width_y_uvec, dot(Psi_3D, width_y_uvec)).imag)
    
    
    # cylindrical integration along the beam
    R_bound = 5 * (width_x_mag + width_y_mag) /2 # average of the two widths
    
    V = omega * g_magnitude / 2.998e8
    rho = np.arange(0, 1, 1 / len_rho)
    P_deposited = np.zeros(len(rho))
    dP_dtau = (2 * gamma * P_along_beam)
    
    for i in range(len(tau)):
        print(i) ## to check progress
        if dP_dtau[i] > 1e-2 / len(tau): # skip insignificant cases
        
            C = 2 * dP_dtau[i] / np.pi / width_x_mag[i] / width_y_mag[i] / V[i] * len_rho
            dP_cross_sect = 0
            P_deposited_step = np.zeros(len(rho))
            
            for j in range(int(R_bound[i] / R_step)):
                R_length = R_step * j
                dV = R_step * (R_length + R_step / 2) * theta_step * dtau[i]
                
                
                for k in range(int(2 * np.pi / theta_step)):
                    theta = theta_step * k
                     
                    
                    W = R_length * (np.cos(theta) * width_x_uvec[i, :] + np.sin(theta) * width_y_uvec[i, :])
                    coord = beam[i, :] + W
                    phi = np.dot(W, np.dot(Psi_3D[i, :, :].imag, W)) / 2
                    
                    
                    
                    dP = C * np.exp(-2 * phi) * dV
                    flux = interp_poloidal_flux(coord[0], coord[2])
                    
                    if flux >= 0:
                        if round(np.sqrt(flux) * len_rho) < len_rho:
                            P_deposited_step[round(np.sqrt(flux) * len_rho)] += dP.real
                            dP_cross_sect += dP.real
            P_deposited = P_deposited + P_deposited_step / dP_cross_sect * dP_dtau[i] * len_rho * dtau[i]
            
    return P_deposited


def heat_me_up(dt, Te_along_beam, m_range, P0, len_rho, R_step, theta_step, 
               figure_flag=True, beam_power_figure_flag=False, Integration_method_flag=False):
    """
    Calculate the power deposition
    
    Overview
    ========
    1. Calculate the hot dielectric tensor from scotty output
        Refer to Felix Parra notes Collisionless Plasma lect VI
    
    2. Calculate the absorption coefficient from antihermitian part of dielectric tensor
        gamma = e_hat* Epsilon_antihermitian e_hat
    
    3. Calculate the total power along beam through ivp solver
    
    4. Two methods available for calculating heat deposition (refer to Torbeam 2.0 paper)):
        a. Gaussian: treat each step as a Gaussian profile and add up their contribution
        to each flux surface, fast and smooth, yet slightly inaccurate at the centre
        
        b. Integration: integration along the beam and add up the power deposition contribution
        from each grid, quite slow but supposedly more accurate
        Usually require len_tau > 1000 len_rho < 500 for smooth results.
        
        Gaussian method is the default output, and the function has the option to output Integration
        method along the Gaussian as comparison
    
    The function outputs two graphs (optional): total power along beam (to identify resonance peaks more easily)
        and dP/d rho against rho (power deposition on flux surfaces). It is different from Torbeam output (dP/dV)
        so extra processing is still required for comparison. The beam path is the dotted black line (normalised_tau
        against rho_along_beam) to guide the interpretation of the graph
    
    Future work:
        1. Relativistic correction is not double checked (I simply rectify all electron masses, should work)
        2. E_hat basis change is not double checked, but O mode and X mode behaviour is as expected
        3. For integration method, maybe some constant is missing, I forcefully scale up the net power contribution
        of each tau_step to be dP_dtau * dtau / drho to solve the problem.
        4. For Gaussian method, currently assumes circular flux surface cross section,
        can choose better approximation or find the exact shape.
        


    Parameters
    ----------
    dt : datatree
        Scotty outputfile
    Te_along_beam : Array
        Temperature value along the beam
    m_range : int
        The range of harmonics calculated in hot dielectric
    P0 : float
        Initial beam power.
    len_rho : int
        Resolution of flux surfaces.
    R_step : float
        Integration step radially outwards from the central beam.
    theta_step : float
        Integration step of the angle around the central beam.
    figure_flag : Bool, optional
        Output power deposition figures
    beam_power_figure_flag : Bool, optional
        The default is False.
        Output power_along_beam figure
    Integration_method_flag: Bool, optional
        The default is False.
        Whether to implement the integration method
    Returns
    -------
    P_deposited_Gaussian : Array
    
    if Integration_method_flag: P_deposited_Gaussian, P_deposited_Integration

    """
    
    # data preparation
    omega = dt.inputs.launch_angular_frequency.data
    tau = dt.analysis.tau.data
    e_hat = dt.analysis.e_hat.data
    # transform the basis of e_hat
    e_hat_transformed = xy_to_u_basis_change(dt, e_hat, "vector")


    # calculation
    Epsi_cold, Epsi_h, Epsi_a = hot_dielectric(
        dt, Te_along_beam, m_range, omega)
    print("Hot dielectrics calculated")
    
    gamma = absorption_coef(e_hat_transformed, Epsi_a).imag
    P_along_beam = power_along_beam(dt, gamma, P0)
    print("Power along beam calculated")
    
    beam_plus_flux, beam_minus_flux = poloidal_flux_periphery(dt)
    P_deposited_Gaussian = power_deposition_Gaussian(
        dt, P_along_beam, gamma, beam_plus_flux, beam_minus_flux, len_rho=len_rho)
    print("Gaussian method calculated")
    
    rho = np.linspace(0, 1, len_rho)
    
    if Integration_method_flag:
        P_deposited_integration = power_deposition_integration(dt, P_along_beam, gamma, R_step=R_step, theta_step=theta_step, len_rho = len_rho)
        print("Integration method calculated")
        print('Integration method total power deposited: ', np.sum(P_deposited_integration) / len(P_deposited_integration))
    
    # check result: full absorption should give 1 for both cases
    print('Gaussian method total power deposited: ', np.sum(P_deposited_Gaussian) / len(P_deposited_Gaussian))
    

    # analysis
    if figure_flag == True:
        plt.figure()
        plt.plot(rho, P_deposited_Gaussian,
                 label='power deposition Gaussian (power againt rho)')
        if Integration_method_flag:
            plt.plot(rho, P_deposited_integration, label='power deposition Integration (power againt rho)')
        plt.plot(np.sqrt(dt.analysis.poloidal_flux.data), tau / tau.max(), '--', label = 'beam trajectory (normalised tau against rho)')
        # plt.plot(np.sqrt(poloidal_flux), B_magnitude / B_magnitude.max(), label = 'beam trajectory (normalised B_magnitude against rho)')
        plt.xlabel("Rho")
        plt.ylabel("dP / drho")
        plt.legend(fontsize=10, loc="upper left")


    if beam_power_figure_flag == True:
        plt.figure()
        plt.plot(tau, P_along_beam, label='Power along the beam')
        plt.plot(tau, gamma, label='Absorption')
        plt.legend(fontsize=10, loc="lower left")
        plt.xlabel("tau")
        plt.ylabel("normalisd power / absorption")

    if Integration_method_flag:
        return P_deposited_Gaussian, P_deposited_integration
    else:
        return P_deposited_Gaussian



# path = "C:\\ihpc_internship\\24summer\\calculation_test"
path = "C:\\ihpc_internship\\24summer\\calculation_test"
dt = datatree.open_datatree(path+"\\scotty_output.h5", engine="h5netcdf")


Te_path = "C:\\ihpc_internship\\24summer\\test\\Te.dat"
temperature_scale = 5 # scale up or down the Te profile for testing
Te_along_beam = Te_along_beam_interp(Te_path, dt) * temperature_scale


m_range = 5
P0 = 1
len_rho = 300
R_step, theta_step = 0.01, 0.1

# If integration_method_flag is true, two outputs
P_deposited = heat_me_up(dt, Te_along_beam, m_range, P0, len_rho, R_step, theta_step,
                         figure_flag=True, beam_power_figure_flag=True,
                         Integration_method_flag=False)

# np.savetxt('test1.txt', P_deposited)
