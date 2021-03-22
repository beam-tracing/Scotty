# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 08:37:48 2020

@author: VH Chen
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy import interpolate as interpolate
import math
from scipy import constants as constants
import tikzplotlib

# Vectorised some of the calculation in this version
# This version does not do some of the calculations, as said calculations have been added to Scotty proper



suffix_O = '_O'
suffix_X = '_X'

loadfile = np.load('data_input' + suffix_O + '.npz')
launch_freq_GHz =loadfile['launch_freq_GHz']
loadfile.close()

loadfile = np.load('analysis_output' + suffix_O + '.npz')
cutoff_index_O = loadfile['cutoff_index']
distance_along_line_O = loadfile['distance_along_line']
loc_p_O = loadfile['loc_p']
loadfile.close()

loadfile = np.load('analysis_output' + suffix_X + '.npz')
cutoff_index_X = loadfile['cutoff_index']
distance_along_line_X = loadfile['distance_along_line']
loc_p_X = loadfile['loc_p']
loadfile.close()

launch_angular_frequency = 2*math.pi*10.0**9 * launch_freq_GHz

plot_every_n_points = 1
out_index_new=len(distance_along_line_O)

constant_coefficient = constants.e**4 / (launch_angular_frequency**2 *constants.epsilon_0*constants.m_e)**2
l_lc_O = distance_along_line_O - distance_along_line_O[cutoff_index_O]
l_lc_X = distance_along_line_X - distance_along_line_X[cutoff_index_X]


plt.figure()
plt.title('O-mode')
plt.plot(l_lc_O[:out_index_new:plot_every_n_points],loc_p_O[:out_index_new:plot_every_n_points]/constant_coefficient,linewidth=4.0)
plt.xlabel(r'$(l - l_c) / m$')
plt.ylabel(r'$ \varepsilon / (e^4 \Omega^{-4} \epsilon_0^{-2} m_e^{-2} ) / m^{-3}$')
plt.ylim(0.99, 1.01)
plt.axvline(0,color='k', linestyle='dashed')
plt.axhline(1.0,color='k', linestyle='dashed')
tikzplotlib.save("loc_p_O.tex")

plt.figure()
plt.title('X-mode')
plt.plot(l_lc_X[:out_index_new:plot_every_n_points],loc_p_X[:out_index_new:plot_every_n_points]/constant_coefficient,linewidth=4.0)
plt.xlabel(r'$(l - l_c) / m$')
plt.ylabel(r'$ \varepsilon / (e^4 \Omega^{-4} \epsilon_0^{-2} m_e^{-2} ) / m^{-3} $')
plt.axvline(0,color='k', linestyle='dashed')
tikzplotlib.save("loc_p_X.tex")