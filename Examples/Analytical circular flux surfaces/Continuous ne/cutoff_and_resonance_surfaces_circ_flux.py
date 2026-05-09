# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 15:40:54 2024

@author: matth
"""


import matplotlib.pyplot as plt
from scotty.profile_fit import QuadraticFit
import numpy as np
import datatree
from scotty.geometry import (
    MagneticField,
    CircularCrossSectionField,
    ConstantCurrentDensityField,
    InterpolatedField,
    CurvySlabField,
    EFITField,
)
from scipy import constants
from math import *

from scotty.plotting import (
    plot_dispersion_relation,
    plot_poloidal_beam_path,
    plot_toroidal_beam_path,
    plot_instrumentation_functions
    )
import h5py
from matplotlib.lines import Line2D



def find_electron_mass(
    temperature=None,
):
    r"""Implements first-order relativistic corrections to electron mass.
    Tmperature is an optional argument. If no argument is passed, returns
    standard electron mass as a scalar. When passed an array of temperatures,
    returns an array of relativistically-corrected electron masses.

    Temperature needs to be in units of KeV.
    """

    if temperature is None:
        # print('electron_mass used is standard value.')
        return constants.m_e

    else:
        mazzu = 1 + temperature * 4.892 * (
            10 ** (-3)
        )  # Mazzucato's relativistic correction
        electron_mass = constants.m_e * mazzu
        # (5/2) / (constants.m_e * constants.c**2) * constants.e = 4.892 * (10 ** (-6))
        # But Te is in KeV not eV
        # print('electron_mass used is: ' + str(electron_mass))
        return electron_mass



path = 'C:/Users/matth/Downloads/Scotty/Examples/Analytical circular flux surfaces/Continuous ne/scotty_output.h5'
dt = datatree.open_datatree(path, engine="h5netcdf")
plot_poloidal_beam_path(dt,zoom=1)
mass = 9.1093837*10**(-31)
epsilon = 8.854187817*10**(-12)
charge = 1.60217663*10**(-19)

# Generating inputs, note that these are only for circular geometry.

neFit = QuadraticFit(poloidal_flux_zero_profile=1,ne_0=1)
TeFit = QuadraticFit(poloidal_flux_zero_profile=1,ne_0=15.3) #set temperature at axis by changing ne_0 argument
    
Rmesh = np.array(dt.inputs.poloidalFlux_grid.coords['R'])
Zmesh =  np.array(dt.inputs.poloidalFlux_grid.coords['Z'])
polflux = np.array(dt.inputs.poloidalFlux_grid)
###Figure out the magnetic field (use scotty code)
fieldGenerator = ConstantCurrentDensityField(float(dt.inputs.B_T_axis), float(dt.inputs.R_axis), float(dt.inputs.minor_radius_a), float(dt.inputs.B_p_a))
B_r_dat = np.zeros((len(Rmesh),len(Zmesh)))
B_z_dat = np.zeros((len(Rmesh),len(Zmesh)))
B_t_dat = np.zeros((len(Rmesh),len(Zmesh)))

for ii in range(len(Rmesh)):
    for jj in range(len(Zmesh)):
        B_r_dat[ii,jj] = fieldGenerator.B_R(Rmesh[ii],Zmesh[jj])
        B_t_dat[ii,jj] = fieldGenerator.B_T(Rmesh[ii],Zmesh[jj])
        B_z_dat[ii,jj] = fieldGenerator.B_Z(Rmesh[ii],Zmesh[jj])




#calculate resonance and cutoff surfaces. Variables with suffix R are the relativistic versions

#electron cyclotron frequency

cyclofreq = np.zeros((len(Rmesh),len(Zmesh)))
cyclofreq_R = np.zeros((len(Rmesh),len(Zmesh)))
for ii in range(len(Rmesh)):
    for jj in range(len(Zmesh)):
        Bmagnitude = ((B_r_dat[ii][jj])**2+(B_t_dat[ii][jj])**2+(B_z_dat[ii][jj])**2)**0.5
        cyclofreq[ii,jj] = charge*Bmagnitude/mass/(2*pi)*10**(-9)
        cyclofreq_R[ii,jj] = charge*Bmagnitude/find_electron_mass(TeFit(polflux[ii,jj]))/(2*pi)*10**(-9)

# plasma frequency
plasmafreq = np.zeros((len(Rmesh),len(Zmesh)))
plasmafreq_R = np.zeros((len(Rmesh),len(Zmesh)))
for i in range(len(Rmesh)):
    for j in range(len(Zmesh)):
        ne_i = neFit(polflux[i,j])
        plasmafreq[i,j] = ( ( ne_i * 10**(19) * charge**2 )/(epsilon * mass) )**0.5 *1/(2*pi) * 10**(-9)
        plasmafreq_R[i,j] = ( ( ne_i * 10**(19) * charge**2 )/(epsilon * find_electron_mass(TeFit(polflux[i,j]))) )**0.5 *1/(2*pi) * 10**(-9)

# Right hand cutoff frequency

Fr_freq = np.zeros((len(Rmesh),len(Zmesh)))
Fr_freq_R = np.zeros((len(Rmesh),len(Zmesh)))

for i in range(len(Rmesh)):
    for j in range(len(Zmesh)):
       
        Fr_freq[i,j] = 1/(2*pi)*0.5*((cyclofreq[i,j]*2*pi*10**9)+( (cyclofreq[i,j]*10**9*2*pi)**2 + 4*(plasmafreq[i,j]*10**9*2*pi)**2 )**0.5)*10**(-9)
        Fr_freq_R[i,j] = 1/(2*pi)*0.5*((cyclofreq_R[i,j]*2*pi*10**9)+( (cyclofreq_R[i,j]*10**9*2*pi)**2 + 4*(plasmafreq_R[i,j]*10**9*2*pi)**2 )**0.5)*10**(-9)


# Upper hybrid resonance

            
uh_freq = np.zeros((len(Rmesh),len(Zmesh)))
uh_freq_R = np.zeros((len(Rmesh),len(Zmesh)))
for i in range(len(Rmesh)):
    for j in range(len(Zmesh)):
        uh_freq[i,j] = (10**(-9))*1/(2*pi)*((cyclofreq[i,j]*2*pi*10**9)**2+ (plasmafreq[i,j]*2*pi*10**9)**2)**0.5
        uh_freq_R[i,j] = (10**(-9))*1/(2*pi)*((cyclofreq_R[i,j]*2*pi*10**9)**2+ (plasmafreq_R[i,j]*2*pi*10**9)**2)**0.5

#plot non-relativistic surfaces

plt.figure()
plt.contour(Rmesh,Zmesh,np.transpose(polflux),levels=np.linspace(0,1,10),colors='black') #plot out flux surfaces
plt.xlabel('R / m')
plt.ylabel('Z / m')

launchfreq = dt.inputs.launch_freq_GHz.values
plt.contour(Rmesh,Zmesh,np.transpose(cyclofreq),levels = [launchfreq],colors='yellow')
plt.contour(Rmesh,Zmesh,np.transpose(cyclofreq),levels = [launchfreq/2],colors='blue')
#plt.contour(Rmesh,Zmesh,np.transpose(plasmafreq),levels = [launchfreq],colors='green')
#plt.contour(Rmesh,Zmesh,np.transpose(Fr_freq),levels = [launchfreq],colors='red')
#plt.contour(Rmesh,Zmesh,np.transpose(uh_freq),levels = [launchfreq],colors='purple')


# manual legend entries
legend_lines = [
    Line2D([0], [0], color='yellow', label='Fundamental electron cyclotron Resonance'),
    Line2D([0], [0], color='blue', label='Second harmonic electron cyclotron Resonance'),
    #Line2D([0], [0], color='green', label='Plasma Cutoff Frequency (O-mode)'),
    #Line2D([0], [0], color='red', label='Right hand cutoff (X-mode)'),
    #Line2D([0], [0], color='purple', label='Upper Hybrid Resonance'),
    Line2D([0], [0], color='black', label='Trajectory')
]


plt.plot(dt.analysis.q_R,dt.analysis.q_Z, color = 'black',label = 'Trajectory') #plot trajectory

#use the following two lines to zoom in and out
plt.xlim(1.1,1.9)
plt.ylim(-0.2,0.2)

plt.title('Non-relativistic cutoffs and resonances')
plt.legend(
    handles=legend_lines,
    loc='upper left',
    bbox_to_anchor=(1.02, 1)
)

#plot relativistic versions
plt.figure()

plt.contour(Rmesh,Zmesh,np.transpose(polflux),levels=np.linspace(0,1,10),colors='black') #plot out flux surfaces
plt.xlabel('R / m')
plt.ylabel('Z / m')

launchfreq = dt.inputs.launch_freq_GHz.values
plt.contour(Rmesh,Zmesh,np.transpose(cyclofreq_R),levels = [launchfreq],colors='yellow')
plt.contour(Rmesh,Zmesh,np.transpose(cyclofreq_R),levels = [launchfreq/2],colors='blue')
#plt.contour(Rmesh,Zmesh,np.transpose(plasmafreq_R),levels = [launchfreq],colors='green')
#plt.contour(Rmesh,Zmesh,np.transpose(Fr_freq_R),levels = [launchfreq],colors='red')
#plt.contour(Rmesh,Zmesh,np.transpose(uh_freq_R),levels = [launchfreq],colors='purple')


# manual legend entries
legend_lines = [
    Line2D([0], [0], color='yellow', label='Fundamental electron cyclotron Resonance'),
    Line2D([0], [0], color='blue', label='Second harmonic electron cyclotron Resonance'),
    #Line2D([0], [0], color='green', label='Plasma Cutoff Frequency (O-mode)'),
    #Line2D([0], [0], color='red', label='Right hand cutoff (X-mode)'),
    #Line2D([0], [0], color='purple', label='Upper Hybrid Resonance'),
    Line2D([0], [0], color='black', label='Trajectory')
]


plt.plot(dt.analysis.q_R,dt.analysis.q_Z, color = 'black',label = 'Trajectory') #plot trajectory

#use the following two lines to zoom in and out
plt.xlim(1.1,1.9)
plt.ylim(-0.2,0.2)


plt.title('Relativistic cutoffs and resonances')
plt.legend(
    handles=legend_lines,
    loc='upper left',
    bbox_to_anchor=(1.02, 1)
)

# Print out calculated absorption position
print('Calculated absorption position')
print('R:',dt.solver_output.q_R.values[-1],'Z:', dt.solver_output.q_Z.values[-1],'zeta:', dt.solver_output.q_zeta.values[-1])