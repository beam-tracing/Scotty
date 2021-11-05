# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 11:43:13 2021

@author: chenv
"""

import numpy as np
import matplotlib.pyplot as plt

years = np.arange(2010,2020)
expenditures = np.array([34.000,30.724,31.498,32.275,31.396,
                         30.642,29.859,31.835,25.061,26.387]) # millions of euros

plt.figure()
plt.plot(years, expenditures,
         marker='D', markeredgecolor='k', markerfacecolor='w',
         linestyle='dashed', color='r')
plt.ylabel('Expenditure / million Euros')
plt.title('French national spending on magnetic confinement fusion')
plt.xticks(years)
plt.xticks(rotation=45, ha='right')


plt.axis('square')