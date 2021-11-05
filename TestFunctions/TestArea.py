# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 14:52:51 2020

@author: VH Chen
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, interpolate


def find_area_points(xs,ys,fraction_wanted):
    # Assume xs sorted in ascending order
    
    if fraction_wanted > 0.5 or fraction_wanted < 0.0:
        print('Invalid value given for fraction')
    x_vals = np.zeros(2)    
    y_vals = np.zeros(2)    
    
    cumulative_ys          = integrate.cumulative_trapezoid(ys,xs,initial=0)
    cumulative_ys_at_y_max = cumulative_ys[ys.argmax()]
    total_ys               = integrate.simps(ys,xs)
    fraction_at_y_max      = cumulative_ys_at_y_max / total_ys

    if (fraction_at_y_max - fraction_wanted) < 0:
        lower_fraction = 0.0
        upper_fraction = 2 * fraction_wanted

    elif (fraction_at_y_max + fraction_wanted) > 1.0:
        lower_fraction = 1 - 2 * fraction_wanted
        upper_fraction = 1.0

    else:
        lower_fraction = fraction_at_y_max - fraction_wanted
        upper_fraction = fraction_at_y_max + fraction_wanted

    interp_x = interpolate.interp1d(cumulative_ys/total_ys, xs, fill_value='extrapolate')
    x_vals[:] = interp_x([lower_fraction,upper_fraction])

    interp_y = interpolate.interp1d(xs, ys, fill_value='extrapolate')
    y_vals[:] = interp_y(x_vals)

    return x_vals, y_vals

def fun_gaussian(xs,mean,stdev):
    return (stdev * np.sqrt(2*np.pi))**(-1) * np.exp(-0.5*((xs-mean)/stdev)**2)


x_array = np.linspace(-1,1,101)
y_array = fun_gaussian(x_array,0.0,0.2)

x_vals, y_vals = find_area_points(x_array,y_array,0.341)