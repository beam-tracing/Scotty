# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 09:43:34 2021

@author: VH Hall-Chen (valerian@hall-chen.com)

This script figures out the Gaussian beam properties based on the horn and lens

Much of the theory is very nicely explained in Goldsmith's Quasioptics. Just
note that what he calls 'flare angle', I call 'semiflare angle'
- Table 7.1 for ratios of beam widths to aperture radii
"""
import numpy as np
from scipy import constants
import sys

def make_my_horn(name):
    if name == 'MAST_V_band':
        myHorn = Scalar_Horn(name) 
        
        FWHM_angle = 25 #deg
        far_field_divergence_angle = np.rad2deg(np.arctan(np.tan(np.deg2rad(FWHM_angle))/np.sqrt(2 * np.log(2))))
        # np.log gives the natural log, which is what I want
        # myHorn.far_field_divergence_angle = FWHM_angle / 1.18 % Estimate, from Goldsmith
        
        # I assume the horn is aperture-limited, so the width at the output is
        # independent of freq
        mid_band_freq = (75.0 + 50.0)*10**9 / 2
        w_0 = constants.c / (np.pi * mid_band_freq * np.tan(np.deg2rad(far_field_divergence_angle)))
        # w_0 = 0.003855191833541916 # This is what the calculation gives me
        # w_0 = 0.004054462688018294 # This is what Neal gets

        myHorn.aperture_radius = w_0 / 0.644
        
    elif name == 'MAST_Q_band':
        aperture_radius = inch2m(1.44/2)
        semiflare_angle = np.deg2rad(30.00/2) #NSTX Q-band horn
        # semiflare_angle = np.deg2rad(16.75/2) #NSTX V-band horn
        
        myHorn = Conical_Horn(name)
        
        myHorn.aperture_radius = aperture_radius
        myHorn.semiflare_angle = semiflare_angle

    elif name == 'MAST_Q_band_alt':
        # Data from DC Speirs's CST simulations
        # Not in used at the moment
        
        # 35 GHz
        FWHM_angle_E = 14.4 #deg
        FWHM_angle_H = 17.4 #deg
        # 40 GHz
        FWHM_angle_E = 12.7 #deg
        FWHM_angle_H = 15.3 #deg
        # 45 GHz
        FWHM_angle_E = 11.5 #deg
        FWHM_angle_H = 13.7 #deg
        # 50 GHz
        FWHM_angle_E = 10.7 #deg
        FWHM_angle_H = 12.4 #deg
        
    return myHorn        

def inch2m(length_inches):
    # Converts inches to meters
    length_m = length_inches*25.4/1000
    return length_m

class Horn:  
      
    # init method or constructor   
    def __init__(self, name):  
        self.name = name   
        
    # Sample Method   
    def who_am_I(self):  
        print('Horn name: ', self.name)  


class Conical_Horn(Horn):
    """
    inherits Horn
    
    Properties
    - smooth walled
    - circular cross-section
    """
    
    def __init__(self, name):
        # Calling init of parent class
        super().__init__(name)
        
        self.aperture_radius = None
        self.semiflare_angle = None
        
        # self.is_symmetric = False # Whether the output beam is symmetric
        # Method for determining the beam width
        self.width_method = 'Goldsmith_symmetric' # Uses Goldsmith's Quasioptics textbook
        # self.width_method = 'Goldsmith_asymmetric' # Uses Goldsmith's Quasioptics textbook
        # self.width_method = 'Speirs_symmetric' # Uses average of David Speirs's CST simulation data
        # self.width_method = 'Speirs_asymmetric' # Uses David Speirs's CST simulation data

        # self.ratio_aperture_width = None # a / w at the aperture
        
    def output_beam(self, freq_GHz=None):
        """
        Gives the width and curvature at the mouth of the horn for a given freq
        Note that this is curvature, the reciprocal of radius of curvature
        """
        if self.aperture_radius is None:
            print('Warning: aperture_radius not initialised')
        elif self.semiflare_angle is None:
            print('Warning: semiflare_angle not initialised')
    
        
        slant_length = self.aperture_radius / np.sin(self.semiflare_angle)

        curvature = 1 / slant_length
        
        if self.width_method == 'Goldsmith_symmetric':
            width = 0.76 * self.aperture_radius 
            
        elif self.width_method == 'Goldsmith_asymmetric':
            # w_E > w_H seems to be the opposite of what Jon measures, though
            # Should pass it through the optics and everything and see what
            # I'd expect him to have measured
            width_E = 0.88 * self.aperture_radius 
            width_H = 0.64 * self.aperture_radius 
            width = [width_E,width_H]

        elif self.width_method == 'Speirs_asymmetric':
            if freq_GHz is None:
                print('Frequency required for this width_method')
                sys.exit()            
            
            # Beam waist widths don't change very much, it seems
            w0_E = 0.0124
            w0_H = 0.01033   
        
            # Calculates the width of the Rayleigh range
            zR_E = np.pi * w0_E**2 * (freq_GHz * (10**(9) / constants.c) )
            zR_H = np.pi * w0_H**2 * (freq_GHz * (10**(9) / constants.c) )
            print(zR_E)
            print(zR_H)
            
            distance_from_w0 = self.aperture_radius / np.tan(self.semiflare_angle)

            width_E = w0_E * np.sqrt(1 + (distance_from_w0/zR_E)**2)
            width_H = w0_H * np.sqrt(1 + (distance_from_w0/zR_H)**2)
            print(width_E/self.aperture_radius)
            print(width_H/self.aperture_radius)
            width = [width_E,width_H]
            
        elif self.width_method == 'Speirs_symmetric':
            if freq_GHz is None:
                print('Frequency required for this width_method')
                sys.exit()
            #Not yet implemented
            
            width_E = 0.88 * self.aperture_radius 
            width_H = 0.64 * self.aperture_radius 
            width = [width_E,width_H]
            
        else:
            print('Invalid width_method')
            sys.exit()
            
        return width, curvature
            
        

        
class Scalar_Horn(Horn):
    """
    inherits Horn
    """
    
    def __init__(self, name):
        # Calling init of parent class
        super().__init__(name)
        
        self.is_aperture_limited = True
        # True: aperture-limited / diffraction-limited
        # A useful approximation

        self.aperture_radius = None


    def output_beam(self,freq=None):
        """
        Gives the width and curvature at the mouth of the horn for a given freq
        Note that this is curvature, the reciprocal of radius of curvature
        """
        if self.is_aperture_limited:
            if self.aperture_radius is not None:
                width = 0.644 * self.aperture_radius
                curvature = 0
                
                
        return width, curvature



















