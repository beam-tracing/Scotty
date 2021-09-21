# -*- coding: utf-8 -*-
"""
lens-a-lot

Created on Sat Apr 17 16:23:01 2021

@author: VH Hall-Chen (valerian@hall-chen.com)

Transfer matrix (2x2):

(   r'   )     ( A   B )  (   r   )
(        )  =  (       )  (       )
( theta' )     ( C   D )  ( theta )

Consider a ray entering and leaving an optical system. It is distance r above
the system axis, entering at an angle theta. It leaves at a distance r', and 
an angle theta'
"""
import numpy as np
from scipy import constants

def make_my_lens(name):
    myLens = Thin_Lens(name)
    
    if name == 'MAST_V_band':
        myLens.focal_length = 0.125
    elif name == 'MAST_Q_band':
        myLens.focal_length = 0.27
        
    return myLens 

class Lens:
    # init method or constructor   
    def __init__(self, name):  
        self.name = name  
        self.focal_length = None
        self.force_thin_lens_approx = True
        
    # Sample Method   
    def who_am_I(self):  
        print('Lens name: ', self.name)
        
    
class Thin_Lens(Lens):
    """
    inherits Lens
    """
    
    def __init__(self, name):
        # Calling init of parent class
        super().__init__(name)
        
    def output_beam(self, Psi_w_in, freq_GHz):
        """
        - Psi_w_in should be a 2x2 matrix
        - Assumes the real part of Psi_w is diagonalised
        - Doesn't assume that the diagnoalised Re(Psi_w)'s components are equal 
          to each other, that is, the curvatures are allowed to be 'elliptical'
        """
        if self.focal_length is None:
            print('Warning: focal length not initialised')
            
        angular_frequency = 2*np.pi*10.0**9 * freq_GHz
        wavenumber = angular_frequency / constants.c
        
        Psi_w_out = Psi_w_in - np.eye(2) * wavenumber / self.focal_length
        
        return Psi_w_out
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        