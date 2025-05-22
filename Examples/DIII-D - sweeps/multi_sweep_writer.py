# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 17:13:01 2024

@author: matth
"""
from matplotlib.ticker import FormatStrFormatter
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from math import *
import scipy
from scipy import interpolate,optimize,fmin
from scipy import constants



def multisweep(pol_launch_angles,tor_launch_angles,launch_freqs_GHz, folder_path,suffix_format):
    #data arrays
    mismatchplot = np.zeros((len(pol_launch_angles),len(tor_launch_angles),len(launch_freqs_GHz)))
    deltamismatch = np.zeros((len(pol_launch_angles),len(tor_launch_angles),len(launch_freqs_GHz)))
    Kcutoff = np.zeros((len(pol_launch_angles),len(tor_launch_angles),len(launch_freqs_GHz)))
    rhocutoff = np.zeros((len(pol_launch_angles),len(tor_launch_angles),len(launch_freqs_GHz)))

    
    
    #now, consolidate all the data
    
    for k in range(len(pol_launch_angles)):
        for i in range(len(tor_launch_angles)):
            for j in range(len(launch_freqs_GHz)):
                outputsuffix = suffix_format.format(
                    pol_launch_angles[k],
                    tor_launch_angles[i],
                    launch_freqs_GHz[j]
                    )
                path = folder_path+'scotty_output'+outputsuffix+".h5"
                dt = xr.open_datatree(path, engine="h5netcdf")
                cutoffindex = int(dt.analysis.cutoff_index)
                delta_theta = float(dt.analysis.delta_theta_m[cutoffindex])
                thetam = float(dt.analysis.theta_m[cutoffindex])
                mismatchplot[k,i,j] = thetam   #kth row, jth column
                deltamismatch[k,i,j] = delta_theta
                Kmag_cut = float(dt.analysis.K_magnitude[cutoffindex])
                rhocut = float((dt.analysis.poloidal_flux[cutoffindex])**0.5)
                Kcutoff[k,i,j] = Kmag_cut
                rhocutoff[k,i,j] = rhocut
                
    
    #interpolate mismatch angle and delta_theta
    loc_m  = np.zeros((len(pol_launch_angles),len(tor_launch_angles),len(launch_freqs_GHz)))
    for k in range(len(pol_launch_angles)):
        interp_mismatch = interpolate.RectBivariateSpline(
            tor_launch_angles,
            launch_freqs_GHz,
            mismatchplot[k,:,:],
            kx=3,
            ky=3,
            s=0
            ) 
        interp_delta = interpolate.RectBivariateSpline(
            tor_launch_angles,
            launch_freqs_GHz,
            deltamismatch[k,:,:],
            kx=3,
            ky=3,
            s=0
            ) 
        
        
        #figure out what toroidal angle gives 0 mismatch for a given frequency
        toroidalzeromismatch = []
        for i in launch_freqs_GHz:
            interp_zero_mismatch_toroidal = interpolate.UnivariateSpline(tor_launch_angles,interp_mismatch(tor_launch_angles,i))
            toroidalzeromismatch.append(interp_zero_mismatch_toroidal.roots()[0])
        
        
        #now, we can start finding loc_m
        for i in range(len(tor_launch_angles)):
            for j in range(len(launch_freqs_GHz)):
                outputsuffix = suffix_format.format(
                    pol_launch_angles[k],
                    tor_launch_angles[i],
                    launch_freqs_GHz[j]
                    )
                path = folder_path+'scotty_output'+outputsuffix+".h5"
                dt = xr.open_datatree(path, engine="h5netcdf")
                delta_theta = interp_delta(toroidalzeromismatch[j],launch_freqs_GHz[j])[0][0] #this gives us delta_theta 
                thetam = float(dt.analysis.theta_m[cutoffindex])
                loc_m[k,i,j] = exp(-2 * thetam**2/ delta_theta**2 )
    
    
    #write data into main file
    outputs = xr.Dataset(
    {
        "mismatch_angle_plot": (["pol_launch_angle", "tor_launch_angle", "frequency"], mismatchplot),
        "mismatch_atten_plot": (["pol_launch_angle", "tor_launch_angle", "frequency"], loc_m),
        "K_cutoff_plot": (["pol_launch_angle", "tor_launch_angle", "frequency"], Kcutoff),
        "rho_cutoff_plot": (["pol_launch_angle", "tor_launch_angle", "frequency"], rhocutoff)

    },
    coords={
        "pol_launch_angle": pol_launch_angles,
        "tor_launch_angle": tor_launch_angles,
        "frequency": launch_freqs_GHz,
    },
)
    
    dt = xr.DataTree.from_dict({"outputs": outputs})
    dt.to_netcdf(
        "sweep_main.h5",
        engine="h5netcdf",
        invalid_netcdf=True,
    )


#parameters you swept
pol_launch_angles = [-8,-11.4]
tor_launch_angles = np.linspace(-4.0, 9.0, 66)
launch_freqs_GHz = np.array([55.0,57.5,60.0,62.5,65.0,67.5,70.0,72.5,75.0])

# where all the .h5 files are stored
folder_path = "C:/Users/matth/Downloads/Scotty/Examples/DIII-D - sweeps/" 

# indicate how the suffix is formatted. numbers before the : are indices, numbers before f is the number of d.p.
suffix_format = '_pol{0:.1f}_tor{1:.1f}_f{2:.1f}_X' 

multisweep(pol_launch_angles,tor_launch_angles,launch_freqs_GHz,folder_path,suffix_format)

