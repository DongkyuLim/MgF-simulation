# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 02:16:58 2022

@author: qmopl
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

wavelength = 359.3e-9 # m to mm
k = 1/wavelength*2*np.pi #x_0
x0 = 1/k


xs = np.linspace(-0.4,0.4,101)/x0
ys = np.linspace(-0.4,0.4,101)/x0
zs = np.linspace(-0.2,0.2,101)/x0

X,Y,Z = np.meshgrid(xs,ys,zs,sparse=1,indexing="ij")
B = np.load("1258_B_field.npy")

print(B)
