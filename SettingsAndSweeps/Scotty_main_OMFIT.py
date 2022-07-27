# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 10:44:34 2018

@author: VH Hall-Chen
Valerian Hongjie Hall-Chen
valerian@hall-chen.com valerian_hall-chen@ihpc.a-star.edu.sg
"""
from Scotty_beam_me_up import beam_me_up
import numpy as np

loadfile = np.load('inScotty.npz', allow_pickle=True)
args_dict = loadfile['args_dict']
kwargs_dict = loadfile['kwargs_dict']
loadfile.close()

beam_me_up(**args_dict.item(), **kwargs_dict.item())
    

