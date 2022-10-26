# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 21:10:53 2020

@author: VH Chen
"""

import numpy as np
import matplotlib.pyplot as plt


def plot():
    loadfile = np.load("data_input3.npz")
    ne_data_density_array = loadfile["ne_data_density_array"]
    ne_data_radialcoord_array = loadfile["ne_data_radialcoord_array"]
    loadfile.close()

    ne_data_poloidal_flux_array = ne_data_radialcoord_array**2

    plt.figure()
    plt.plot(ne_data_poloidal_flux_array, ne_data_density_array)
    plt.ylabel("density / 1e19 m-3")


if __name__ == "__main__":
    plot()
