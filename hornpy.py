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
        # w_0 = 0.004054462688018294 # This is what Neal gets

        myHorn.aperture_radius = w_0 / 0.644
        
    elif name == 'MAST_Q_band':
        aperture_radius = inch2m(1.44/2)
        semiflare_angle = np.deg2rad(30.00/2)
        # semiflare_angle = np.deg2rad(16.75/2) #NSTX V-band horn
        
        myHorn = Conical_Horn(name)
        
        myHorn.aperture_radius = aperture_radius
        myHorn.semiflare_angle = semiflare_angle

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
        
        self.is_symmetric = True # Whether the output beam is symmetric

        # self.ratio_aperture_width = None # a / w at the aperture
        
    def output_beam(self):
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
        
        if self.is_symmetric:
            width = 0.76 * self.aperture_radius 
            
        else:
            # w_E > w_H seems to be the opposite of what Jon measures, though
            # Should pass it through the optics and everything and see what
            # I'd expect him to have measured
            width_E = 0.88 * self.aperture_radius 
            width_H = 0.64 * self.aperture_radius 
            width = [width_E,width_H]
            
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



















