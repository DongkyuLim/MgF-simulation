# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 21:06:26 2021

@author: qmopl
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as cts
import pylcp
import time
import random
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy import stats
from scipy import integrate
from scipy import special

class Whitelight:
    '''
    Useful module that calculate main-laser slowing beam`s motion trace
    
    Parameters
    ----------
    main_det : float or int
        Frequency detune of main slowing beam
    chirp_coeff : float or int
        Change of frequency detune in 1000000 $t_0$(about 7.2 ms)
    power_rate : float or int
        Power rate between main beam and each sideband.
        For example, power_rate = 1.2 means that main laser`s power is 1.2 times stronger than one sideband`s power.
    laseron : float or int
        Time when the laser switch is on.
    laseroff : float or int
        Time when the laser switch is off.
    
    '''
    def __init__(self, main_det,det_1,det_2,beta_1,beta_2,laseron,laseroff):
        self.main_det = main_det
        self.det_1 = det_1
        self.det_2 = det_2
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.sols = list()
        self.v_trap_initial = list()
        self.time_final = list()
        self.laseron = laseron
        self.laseroff = laseroff
        

        
        
        #Define the constants
        #It does not change
        Gamma = 22 # Hz to kHz, Decay rate
        wavelength = 359.3e-9 # m to mm
        k = 1/wavelength*2*np.pi #x_0
        x0 = 1/k
        t0 = 1/Gamma*1/(2*np.pi*1e6)
        v0 = x0/t0
        m0 = cts.hbar*t0/x0**2
        a0 = x0/t0**2
        F0 = cts.hbar/(x0*t0)
        mass = 43*cts.value('atomic mass constant')/m0 # m_0
        #It can change 
        waist = 0.012/x0
        v_max = 20
        z_max = 384.855e-3/x0
        dz = 0.05/x0
        dv = 0.05
        omega = 2*np.pi*(cts.c/wavelength) #Transition frequency
        Isat = np.pi*cts.h*cts.c*Gamma*2*np.pi*1e6/3*1/(wavelength)**3
        
        self.zz = np.linspace(-1*z_max,z_max,100)
        self.vv = np.linspace(-1*v_max,v_max,100)
        self.Z, self.V = np.meshgrid(self.zz,self.vv)
        self.t0 = t0
        self.v0 = v0
        self.x0 = x0
        self.a0 = a0
        self.m0 = m0
        self.F0 = F0
        self.mass = mass
        self.k = k
        self.wavelength = wavelength
        self.Gamma = Gamma
        self.omega = omega
        self.Isat = Isat
        self.mag_field_grad = 1252.8168984164048*x0
        

        # The detunings used in the PRAs:
        intensity = 2./(np.pi*(0.012)**2)/Isat


        #Define the hamiltonian
        H0_X, Bq_X, U_X, Xbasis = pylcp.hamiltonians.XFmolecules.Xstate(B=0,
        gamma = 50.697/Gamma,b=154.7/Gamma, c=178.5/Gamma,
            muB = cts.value('Bohr magneton in Hz/T')/1e6*1e-4/Gamma,return_basis=True
            )

        # b : SI coupling(isotropic), c : Iz Sz coupling(anisotropic), cc : I N coupling, gamma : S N coupling

        E_X = np.unique(np.diag(H0_X))

        H0_A, Bq_A, Abasis = pylcp.hamiltonians.XFmolecules.Astate(
            P=+1, Ahfs=-1.5/Gamma, q=0, p=0,gJ=-0.00002,
            muB=cts.value('Bohr magneton in Hz/T')/1e6*1e-4/Gamma, return_basis=True
            )

        # gJ : Lande g-factor, p : parity(e parity)

        E_A = np.unique(np.diag(H0_A))

        dijq = pylcp.hamiltonians.XFmolecules.dipoleXandAstates(
            Xbasis, Abasis, UX=U_X
            )

        hamiltonian = pylcp.hamiltonian(H0_X, H0_A, Bq_X, Bq_A, dijq,mass = mass)
        
        self.hamiltonian = hamiltonian
        
        def Coil_field(I,R:np.array):
            n = 510
            s = 0.1016 # in meter
            rad = 0.071976 # in meter
            def dBx(theta,L):
                dl = np.array([-rad*np.sin(theta),rad*np.cos(theta),0])
                rprime = R*x0 - np.array([0,0,L])+np.array([rad*np.cos(theta),rad*np.sin(theta),0])
                dB = cts.mu_0/(4*np.pi)*np.cross(dl,rprime)/((np.sum(rprime**2))**(3/2))*I
                # print(dl,rprime,dB)
                return dB[0]
            def dBy(theta,L):
                dl = np.array([-rad*np.sin(theta),rad*np.cos(theta),0])
                rprime = R*x0 - np.array([0,0,L])+np.array([rad*np.cos(theta),rad*np.sin(theta),0])
                dB = cts.mu_0/(4*np.pi)*np.cross(dl,rprime)/((np.sum(rprime**2))**(3/2))*I
                return dB[1]

            def dBz(theta,L):
                dl = np.array([-rad*np.sin(theta),rad*np.cos(theta),0])
                rprime = R*x0 - np.array([0,0,L])+np.array([rad*np.cos(theta),rad*np.sin(theta),0])
                dB = cts.mu_0/(4*np.pi)*np.cross(dl,rprime)/((np.sum(rprime**2))**(3/2))*I
                return dB[2]


            Bx = integrate.quad(dBx,0,2*np.pi,args=(-s))[0]-integrate.quad(dBx,0,2*np.pi,args=(s))[0]
            By = integrate.quad(dBy,0,2*np.pi,args=(-s))[0]-integrate.quad(dBy,0,2*np.pi,args=(s))[0]
            Bz = integrate.quad(dBz,0,2*np.pi,args=(-s))[0]-integrate.quad(dBz,0,2*np.pi,args=(s))[0]

            return np.array([Bx,By,Bz])*n*10000 # Return in Gauss
        
        self.magField = lambda R,t : Coil_field(3.5,R)
        
        
        def Fixed_detune_MgF_MOT(main_det,det_1,det_2,beta_1,beta_2,laseron,laseroff):
            det_side = det_1/Gamma
            det_side2 = det_2/Gamma
            Avg_X = np.average(E_X)
            init_pow = 0.5*2./(np.pi*(0.012)**2)/Isat

            def Gaussain_Beam_Diagonal(R:np.array,waist):
                return np.exp(-2*((R[0]-R[1])**2/2+R[2]**2)/waist**2)

            def Bessel_Intensity(n_order,beta):

                return special.jv(n_order,beta)**2

            def Heav_step(t):
                if laseron<=t and t<laseron+14:
                    return -1*(t-laseron-7)*((t-laseron-7)**2-49*3)*1/686*1/2+1/2
                elif laseron+14<=t and t<laseroff:
                    return 1
                elif t>=laseroff and t<laseroff+14:
                    return (t-laseroff-7)*((t-laseroff-7)**2-49*3)*1/686*1/2 + 1/2
                else:
                    return 0    

            laserBeams = pylcp.laserBeams()

            laserBeams += pylcp.laserBeams([{'kvec':np.array([-1,-1,0]),'pol':+1,'pol_coord':'spherical','delta':(E_A[-1]-Avg_X-main_det)-det_side-det_side2*2,
                                             's': lambda R,t : init_pow*Gaussain_Beam_Diagonal(R,waist)*Heav_step(t)*Bessel_Intensity(-1,beta_1)*Bessel_Intensity(2,beta_2)},
                                            {'kvec':np.array([-1,-1,0]),'pol':-1,'pol_coord':'spherical','delta':(E_A[-1]-Avg_X-main_det)-det_side-det_side2*2,
                                             's': lambda R,t : init_pow*Gaussain_Beam_Diagonal(R,waist)*Heav_step(t)*Bessel_Intensity(-1,beta_1)*Bessel_Intensity(2,beta_2)}])
            laserBeams += pylcp.laserBeams([{'kvec':np.array([-1,-1,0]),'pol':+1,'pol_coord':'spherical','delta':(E_A[-1]-Avg_X-main_det)-det_side-det_side2*1,
                                             's': lambda R,t : init_pow*Gaussain_Beam_Diagonal(R,waist)*Heav_step(t)*Bessel_Intensity(-1,beta_1)*Bessel_Intensity(1,beta_2)},
                                            {'kvec':np.array([-1,-1,0]),'pol':-1,'pol_coord':'spherical','delta':(E_A[-1]-Avg_X-main_det)-det_side-det_side2*1,
                                             's': lambda R,t : init_pow*Gaussain_Beam_Diagonal(R,waist)*Heav_step(t)*Bessel_Intensity(-1,beta_1)*Bessel_Intensity(1,beta_2)}])
            laserBeams += pylcp.laserBeams([{'kvec':np.array([-1,-1,0]),'pol':+1,'pol_coord':'spherical','delta':(E_A[-1]-Avg_X-main_det)-det_side-det_side2*0,
                                             's': lambda R,t : init_pow*Gaussain_Beam_Diagonal(R,waist)*Heav_step(t)*Bessel_Intensity(-1,beta_1)*Bessel_Intensity(0,beta_2)},
                                            {'kvec':np.array([-1,-1,0]),'pol':-1,'pol_coord':'spherical','delta':(E_A[-1]-Avg_X-main_det)-det_side-det_side2*0,
                                             's': lambda R,t : init_pow*Gaussain_Beam_Diagonal(R,waist)*Heav_step(t)*Bessel_Intensity(-1,beta_1)*Bessel_Intensity(0,beta_2)}])
            laserBeams += pylcp.laserBeams([{'kvec':np.array([-1,-1,0]),'pol':+1,'pol_coord':'spherical','delta':(E_A[-1]-Avg_X-main_det)-det_side+det_side2*1,
                                             's': lambda R,t : init_pow*Gaussain_Beam_Diagonal(R,waist)*Heav_step(t)*Bessel_Intensity(-1,beta_1)*Bessel_Intensity(-1,beta_2)},
                                            {'kvec':np.array([-1,-1,0]),'pol':-1,'pol_coord':'spherical','delta':(E_A[-1]-Avg_X-main_det)-det_side+det_side2*1,
                                             's': lambda R,t : init_pow*Gaussain_Beam_Diagonal(R,waist)*Heav_step(t)*Bessel_Intensity(-1,beta_1)*Bessel_Intensity(-1,beta_2)}])
            laserBeams += pylcp.laserBeams([{'kvec':np.array([-1,-1,0]),'pol':+1,'pol_coord':'spherical','delta':(E_A[-1]-Avg_X-main_det)-det_side+det_side2*2,
                                             's': lambda R,t : init_pow*Gaussain_Beam_Diagonal(R,waist)*Heav_step(t)*Bessel_Intensity(-1,beta_1)*Bessel_Intensity(-2,beta_2)},
                                            {'kvec':np.array([-1,-1,0]),'pol':-1,'pol_coord':'spherical','delta':(E_A[-1]-Avg_X-main_det)-det_side+det_side2*2,
                                             's': lambda R,t : init_pow*Gaussain_Beam_Diagonal(R,waist)*Heav_step(t)*Bessel_Intensity(-1,beta_1)*Bessel_Intensity(-2,beta_2)}])

        # Main Slowing Laser
            laserBeams += pylcp.laserBeams([{'kvec':np.array([-1,-1,0]),'pol':+1,'pol_coord':'spherical','delta':(E_A[-1]-Avg_X-main_det)-det_side2*2,
                                             's': lambda R,t : init_pow*Gaussain_Beam_Diagonal(R,waist)*Heav_step(t)*Bessel_Intensity(0,beta_1)*Bessel_Intensity(2,beta_2)},
                                            {'kvec':np.array([-1,-1,0]),'pol':-1,'pol_coord':'spherical','delta':(E_A[-1]-Avg_X-main_det)-det_side2*2,
                                             's': lambda R,t : init_pow*Gaussain_Beam_Diagonal(R,waist)*Heav_step(t)*Bessel_Intensity(0,beta_1)*Bessel_Intensity(2,beta_2)}])
            laserBeams += pylcp.laserBeams([{'kvec':np.array([-1,-1,0]),'pol':+1,'pol_coord':'spherical','delta':(E_A[-1]-Avg_X-main_det)-det_side2*1,
                                             's': lambda R,t : init_pow*Gaussain_Beam_Diagonal(R,waist)*Heav_step(t)*Bessel_Intensity(0,beta_1)*Bessel_Intensity(1,beta_2)},
                                            {'kvec':np.array([-1,-1,0]),'pol':-1,'pol_coord':'spherical','delta':(E_A[-1]-Avg_X-main_det)-det_side2*1,
                                             's': lambda R,t : init_pow*Gaussain_Beam_Diagonal(R,waist)*Heav_step(t)*Bessel_Intensity(0,beta_1)*Bessel_Intensity(1,beta_2)}])
            laserBeams += pylcp.laserBeams([{'kvec':np.array([-1,-1,0]),'pol':+1,'pol_coord':'spherical','delta':(E_A[-1]-Avg_X-main_det)-det_side2*0,
                                             's': lambda R,t : init_pow*Gaussain_Beam_Diagonal(R,waist)*Heav_step(t)*Bessel_Intensity(0,beta_1)*Bessel_Intensity(0,beta_2)},
                                            {'kvec':np.array([-1,-1,0]),'pol':-1,'pol_coord':'spherical','delta':(E_A[-1]-Avg_X-main_det)-det_side2*0,
                                             's': lambda R,t : init_pow*Gaussain_Beam_Diagonal(R,waist)*Heav_step(t)*Bessel_Intensity(0,beta_1)*Bessel_Intensity(0,beta_2)}])
            laserBeams += pylcp.laserBeams([{'kvec':np.array([-1,-1,0]),'pol':+1,'pol_coord':'spherical','delta':(E_A[-1]-Avg_X-main_det)+det_side2*1,
                                             's': lambda R,t : init_pow*Gaussain_Beam_Diagonal(R,waist)*Heav_step(t)*Bessel_Intensity(0,beta_1)*Bessel_Intensity(-1,beta_2)},
                                            {'kvec':np.array([-1,-1,0]),'pol':-1,'pol_coord':'spherical','delta':(E_A[-1]-Avg_X-main_det)+det_side2*1,
                                             's': lambda R,t : init_pow*Gaussain_Beam_Diagonal(R,waist)*Heav_step(t)*Bessel_Intensity(0,beta_1)*Bessel_Intensity(-1,beta_2)}])
            laserBeams += pylcp.laserBeams([{'kvec':np.array([-1,-1,0]),'pol':+1,'pol_coord':'spherical','delta':(E_A[-1]-Avg_X-main_det)+det_side2*2,
                                             's': lambda R,t : init_pow*Gaussain_Beam_Diagonal(R,waist)*Heav_step(t)*Bessel_Intensity(0,beta_1)*Bessel_Intensity(-2,beta_2)},
                                            {'kvec':np.array([-1,-1,0]),'pol':-1,'pol_coord':'spherical','delta':(E_A[-1]-Avg_X-main_det)+det_side2*2,
                                             's': lambda R,t : init_pow*Gaussain_Beam_Diagonal(R,waist)*Heav_step(t)*Bessel_Intensity(0,beta_1)*Bessel_Intensity(-2,beta_2)}])
        # Plus Sideband part
            laserBeams += pylcp.laserBeams([{'kvec':np.array([-1,-1,0]),'pol':+1,'pol_coord':'spherical','delta':(E_A[-1]-Avg_X-main_det)+det_side-det_side2*2,
                                             's': lambda R,t : init_pow*Gaussain_Beam_Diagonal(R,waist)*Heav_step(t)*Bessel_Intensity(1,beta_1)*Bessel_Intensity(2,beta_2)},
                                            {'kvec':np.array([-1,-1,0]),'pol':-1,'pol_coord':'spherical','delta':(E_A[-1]-Avg_X-main_det)+det_side-det_side2*2,
                                             's': lambda R,t : init_pow*Gaussain_Beam_Diagonal(R,waist)*Heav_step(t)*Bessel_Intensity(1,beta_1)*Bessel_Intensity(2,beta_2)}])
            laserBeams += pylcp.laserBeams([{'kvec':np.array([-1,-1,0]),'pol':+1,'pol_coord':'spherical','delta':(E_A[-1]-Avg_X-main_det)+det_side-det_side2*1,
                                             's': lambda R,t : init_pow*Gaussain_Beam_Diagonal(R,waist)*Heav_step(t)*Bessel_Intensity(1,beta_1)*Bessel_Intensity(1,beta_2)},
                                            {'kvec':np.array([-1,-1,0]),'pol':-1,'pol_coord':'spherical','delta':(E_A[-1]-Avg_X-main_det)+det_side-det_side2*1,
                                             's': lambda R,t : init_pow*Gaussain_Beam_Diagonal(R,waist)*Heav_step(t)*Bessel_Intensity(1,beta_1)*Bessel_Intensity(1,beta_2)}])
            laserBeams += pylcp.laserBeams([{'kvec':np.array([-1,-1,0]),'pol':+1,'pol_coord':'spherical','delta':(E_A[-1]-Avg_X-main_det)+det_side-det_side2*0,
                                             's': lambda R,t : init_pow*Gaussain_Beam_Diagonal(R,waist)*Heav_step(t)*Bessel_Intensity(1,beta_1)*Bessel_Intensity(0,beta_2)},
                                            {'kvec':np.array([-1,-1,0]),'pol':-1,'pol_coord':'spherical','delta':(E_A[-1]-Avg_X-main_det)+det_side-det_side2*0,
                                             's': lambda R,t : init_pow*Gaussain_Beam_Diagonal(R,waist)*Heav_step(t)*Bessel_Intensity(1,beta_1)*Bessel_Intensity(0,beta_2)}])
            laserBeams += pylcp.laserBeams([{'kvec':np.array([-1,-1,0]),'pol':+1,'pol_coord':'spherical','delta':(E_A[-1]-Avg_X-main_det)+det_side+det_side2*1,
                                             's': lambda R,t : init_pow*Gaussain_Beam_Diagonal(R,waist)*Heav_step(t)*Bessel_Intensity(1,beta_1)*Bessel_Intensity(-1,beta_2)},
                                            {'kvec':np.array([-1,-1,0]),'pol':-1,'pol_coord':'spherical','delta':(E_A[-1]-Avg_X-main_det)+det_side+det_side2*1,
                                             's': lambda R,t : init_pow*Gaussain_Beam_Diagonal(R,waist)*Heav_step(t)*Bessel_Intensity(1,beta_1)*Bessel_Intensity(-1,beta_2)}])
            laserBeams += pylcp.laserBeams([{'kvec':np.array([-1,-1,0]),'pol':+1,'pol_coord':'spherical','delta':(E_A[-1]-Avg_X-main_det)+det_side+det_side2*2,
                                             's': lambda R,t : init_pow*Gaussain_Beam_Diagonal(R,waist)*Heav_step(t)*Bessel_Intensity(1,beta_1)*Bessel_Intensity(-2,beta_2)},
                                            {'kvec':np.array([-1,-1,0]),'pol':-1,'pol_coord':'spherical','delta':(E_A[-1]-Avg_X-main_det)+det_side+det_side2*2,
                                             's': lambda R,t : init_pow*Gaussain_Beam_Diagonal(R,waist)*Heav_step(t)*Bessel_Intensity(1,beta_1)*Bessel_Intensity(-2,beta_2)}])

#             def MOT_step(t):
#                 if laseroff<=t and t<laseroff+14:
#                     return -1*(t-laseroff-7)*((t-laseroff-7)**2-49*3)*1/686*1/2+1/2
#                 elif laseroff+14<=t:
#                     return 1
#                 else:
#                     return 0     


#             for ii, Eg_i in enumerate(E_X):
#                 if ii==0:
#                     laserBeams += pylcp.laserBeams([{'kvec':np.array([1,0,0]),'pol':-1*pol1,'delta':(E_A[-1]-Eg_i)+d1,
#                                                      's':lambda R,t : s*np.exp(-2*(np.sum(R**2)-R[0]**2)/waist**2)*MOT_step(t)}])
#                     laserBeams += pylcp.laserBeams([{'kvec':np.array([-1,0,0]),'pol':-1*pol1,'delta':(E_A[-1]-Eg_i)+d1,
#                                                      's':lambda R,t : s*np.exp(-2*(np.sum(R**2)-R[0]**2)/waist**2)*MOT_step(t)}])
#                     laserBeams += pylcp.laserBeams([{'kvec':np.array([0,1,0]),'pol':-1*pol1,'delta':(E_A[-1]-Eg_i)+d1,
#                                                      's':lambda R,t : s*np.exp(-2*(np.sum(R**2)-R[1]**2)/waist**2)*MOT_step(t)}])
#                     laserBeams += pylcp.laserBeams([{'kvec':np.array([0,-1,0]),'pol':-1*pol1,'delta':(E_A[-1]-Eg_i)+d1,
#                                                      's':lambda R,t : s*np.exp(-2*(np.sum(R**2)-R[1]**2)/waist**2)*MOT_step(t)}])
#                     laserBeams += pylcp.laserBeams([{'kvec':np.array([0,0,1]),'pol':1*pol1,'delta':(E_A[-1]-Eg_i)+d1,
#                                                      's':lambda R,t : s*np.exp(-2*(np.sum(R**2)-R[2]**2)/waist**2)*MOT_step(t)}])
#                     laserBeams += pylcp.laserBeams([{'kvec':np.array([0,0,-1]),'pol':1*pol1,'delta':(E_A[-1]-Eg_i)+d1,
#                                                      's':lambda R,t : s*np.exp(-2*(np.sum(R**2)-R[2]**2)/waist**2)*MOT_step(t)}])
#                 elif ii==1:
#                     laserBeams += pylcp.laserBeams([{'kvec':np.array([1,0,0]),'pol':-1*pol2,'delta':(E_A[-1]-Eg_i)+d2,
#                                                      's':lambda R,t : s*np.exp(-2*(np.sum(R**2)-R[0]**2)/waist**2)*MOT_step(t)}])
#                     laserBeams += pylcp.laserBeams([{'kvec':np.array([-1,0,0]),'pol':-1*pol2,'delta':(E_A[-1]-Eg_i)+d2,
#                                                      's':lambda R,t : s*np.exp(-2*(np.sum(R**2)-R[0]**2)/waist**2)*MOT_step(t)}])
#                     laserBeams += pylcp.laserBeams([{'kvec':np.array([0,1,0]),'pol':-1*pol2,'delta':(E_A[-1]-Eg_i)+d2,
#                                                      's':lambda R,t : s*np.exp(-2*(np.sum(R**2)-R[1]**2)/waist**2)*MOT_step(t)}])
#                     laserBeams += pylcp.laserBeams([{'kvec':np.array([0,-1,0]),'pol':-1*pol2,'delta':(E_A[-1]-Eg_i)+d2,
#                                                      's':lambda R,t : s*np.exp(-2*(np.sum(R**2)-R[1]**2)/waist**2)*MOT_step(t)}])
#                     laserBeams += pylcp.laserBeams([{'kvec':np.array([0,0,1]),'pol':1*pol2,'delta':(E_A[-1]-Eg_i)+d2,
#                                                      's':lambda R,t : s*np.exp(-2*(np.sum(R**2)-R[2]**2)/waist**2)*MOT_step(t)}])
#                     laserBeams += pylcp.laserBeams([{'kvec':np.array([0,0,-1]),'pol':1*pol2,'delta':(E_A[-1]-Eg_i)+d2,
#                                                      's':lambda R,t : s*np.exp(-2*(np.sum(R**2)-R[2]**2)/waist**2)*MOT_step(t)}]) 
#                 elif ii==2:
#                     laserBeams += pylcp.laserBeams([{'kvec':np.array([1,0,0]),'pol':-1*pol3,'delta':(E_A[-1]-Eg_i)+d3,
#                                                      's':lambda R,t : s*np.exp(-2*(np.sum(R**2)-R[0]**2)/waist**2)*MOT_step(t)}])
#                     laserBeams += pylcp.laserBeams([{'kvec':np.array([-1,0,0]),'pol':-1*pol3,'delta':(E_A[-1]-Eg_i)+d3,
#                                                      's':lambda R,t : s*np.exp(-2*(np.sum(R**2)-R[0]**2)/waist**2)*MOT_step(t)}])
#                     laserBeams += pylcp.laserBeams([{'kvec':np.array([0,1,0]),'pol':-1*pol3,'delta':(E_A[-1]-Eg_i)+d3,
#                                                      's':lambda R,t : s*np.exp(-2*(np.sum(R**2)-R[1]**2)/waist**2)*MOT_step(t)}])
#                     laserBeams += pylcp.laserBeams([{'kvec':np.array([0,-1,0]),'pol':-1*pol3,'delta':(E_A[-1]-Eg_i)+d3,
#                                                      's':lambda R,t : s*np.exp(-2*(np.sum(R**2)-R[1]**2)/waist**2)*MOT_step(t)}])
#                     laserBeams += pylcp.laserBeams([{'kvec':np.array([0,0,1]),'pol':1*pol3,'delta':(E_A[-1]-Eg_i)+d3,
#                                                      's':lambda R,t : s*np.exp(-2*(np.sum(R**2)-R[2]**2)/waist**2)*MOT_step(t)}])
#                     laserBeams += pylcp.laserBeams([{'kvec':np.array([0,0,-1]),'pol':1*pol3,'delta':(E_A[-1]-Eg_i)+d3,
#                                                      's':lambda R,t : s*np.exp(-2*(np.sum(R**2)-R[2]**2)/waist**2)*MOT_step(t)}])
#                 else:
#                     laserBeams += pylcp.laserBeams([{'kvec':np.array([1,0,0]),'pol':-1*pol4,'delta':(E_A[-1]-Eg_i)+d4,
#                                                      's':lambda R,t : s*np.exp(-2*(np.sum(R**2)-R[0]**2)/waist**2)*MOT_step(t)}])
#                     laserBeams += pylcp.laserBeams([{'kvec':np.array([-1,0,0]),'pol':-1*pol4,'delta':(E_A[-1]-Eg_i)+d4,
#                                                      's':lambda R,t : s*np.exp(-2*(np.sum(R**2)-R[0]**2)/waist**2)*MOT_step(t)}])
#                     laserBeams += pylcp.laserBeams([{'kvec':np.array([0,1,0]),'pol':-1*pol4,'delta':(E_A[-1]-Eg_i)+d4,
#                                                      's':lambda R,t : s*np.exp(-2*(np.sum(R**2)-R[1]**2)/waist**2)*MOT_step(t)}])
#                     laserBeams += pylcp.laserBeams([{'kvec':np.array([0,-1,0]),'pol':-1*pol4,'delta':(E_A[-1]-Eg_i)+d4,
#                                                      's':lambda R,t : s*np.exp(-2*(np.sum(R**2)-R[1]**2)/waist**2)*MOT_step(t)}])
#                     laserBeams += pylcp.laserBeams([{'kvec':np.array([0,0,1]),'pol':1*pol4,'delta':(E_A[-1]-Eg_i)+d4,
#                                                      's':lambda R,t : s*np.exp(-2*(np.sum(R**2)-R[2]**2)/waist**2)*MOT_step(t)}])
#                     laserBeams += pylcp.laserBeams([{'kvec':np.array([0,0,-1]),'pol':1*pol4,'delta':(E_A[-1]-Eg_i)+d4,
#                                                      's':lambda R,t : s*np.exp(-2*(np.sum(R**2)-R[2]**2)/waist**2)*MOT_step(t)}])

            return laserBeams
        
        self.laserBeams = Fixed_detune_MgF_MOT(self.main_det,self.det_1,self.det_2,self.beta_1,self.beta_2,self.laseron,self.laseroff)
        
        self.rateeq = pylcp.rateeq(self.laserBeams,self.magField,self.hamiltonian,include_mag_forces=0)
        
    def motion_trace(self,vparam=np.linspace(14,22,9),tparam=np.arange(0,3500000,1)):
        '''

        Parameters
        ----------
        vparam : list with 3 indexes or np.array with shape (3,)
            It determines list of initial velocity when calculating motion trace.
            If vparam = [1,5,0.5], then list of initial velocity is np.arange(1,5,0.5)
        tparam : list with 3 indexes or np.array with shape (3,)
            It determines total time steps that will be evaluated at motion trace calculation.
            If tparam = [1,100,101], then total time steps is np.linspace(1,100,101)

        Returns
        -------
        None, it just saves motion trace solutions at self.sols

        '''

        def Capture_velocity_condition(t,y,threshold = 5.810):
            if y[-6]<threshold:
                val = -1.
            else:
                val = 1.
            return val
        def Lost_condition(t,y,threshold = 0.):
            if y[-6]<threshold:
                val = -1.
            else:
                val = 1.
            return val
        def for_transverse_condition(t,y,threshold = -0.020/self.x0):
            if y[-3]>threshold:
                val = -1.
            else:
                val = 1.
            return val

        Capture_velocity_condition.terminal = False
        Lost_condition.terminal = False
        for_transverse_condition.terminal = False
        conditions =  [for_transverse_condition,Lost_condition,Capture_velocity_condition]

        self.v_longitudinal = np.linspace(14,21,16)
        self.t_eval = np.arange(0,3500000,1)

        for v0_longitudinal in self.v_longitudinal:
            self.rateeq.set_initial_position_and_velocity(np.array([-1*self.zz.max()/np.sqrt(2),-1*self.zz.max()/np.sqrt(2),0]),np.array([v0_longitudinal/np.sqrt(2),v0_longitudinal/np.sqrt(2),0]))
            self.rateeq.set_initial_pop(np.array([1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0]))

            self.rateeq.evolve_motion([0.,max(self.t_eval)],t_eval=self.t_eval,events= conditions,max_step=2e5,progress_bar = 1,method='LSODA')
            sol = self.rateeq.sol
            self.sols.append(sol)
            # print(sol.t_events)
            if len(sol.t_events[0])==1:
                if len(sol.t_events[1])==0:
                    if len(sol.t_events[2])==1:
                        self.time_final.append(sol.t_events[0][0])
                        self.v_trap_initial.append(sol.v[0][0])
            

            

    def plot(self,save=False,save_name=None):
        '''
        

        Parameters
        ----------
        save : True of False, optional
            Whether you want to save graph. The default is False.
        save_name : string if save == True
            File`s name when saving the figure. The default is None.

        Returns
        -------
        None.

        '''

        fig, ax = plt.subplots(1, 1)
        ax.set_xlabel('$X\ (\mathrm{mm})$')
        ax.set_ylabel('$v\ (\mathrm{m/s})$')
        ax.set_title('X axis motion trace')
        ax.set_xlim(-1*self.zz.max()*self.x0*1000,self.zz.max()*self.x0*1000)
        ax.set_ylim(-1*self.v_longitudinal.max()*self.v0,self.v_longitudinal.max()*self.v0)
        fig.subplots_adjust(left=0.12,right=0.9)

        for sol in self.sols:
            ax.plot(sol.r[0]*self.x0*1000,sol.v[0]*self.v0, 'b')
        
        if save:
            fig.savefig(save_name)
        
    def percentage(self):
        '''
        Calculate the percent that satisfies both captured_condition and transverse velocity condition,
        with given main_det, chirp_coeff, power_rate.

        Returns
        -------
        Population : float
            Population that satisfies both captured_condition and transverse velocity condition.

        '''
        if len(self.time_final)<=1:
            return 0.
        def time_vs_v_final(vff):
            if vff>= min(self.v_trap_initial) and vff<=max(self.v_trap_initial):
                return np.interp(vff,self.v_trap_initial,self.time_final)
            
        def v_transverse_vs_v_longitudinal(vfs):
            return (6.00/time_vs_v_final(vfs)*1e-3/self.t0)
        
        muf = 140/self.v0
        sigf = 17/self.v0
        mut = 0
        sigt = 18.7564/self.v0
        
        def Transverse_percentage(v):
            rv = stats.norm(mut,sigt) # Gaussian of transverse, v0 scale
            val = (rv.cdf(v_transverse_vs_v_longitudinal(v))-rv.cdf(0))/rv.cdf(sigt*2)*2
            
            if val >= 1:
                return 1
            else:
                return val
        
        def total_func(v):
            rv_f = stats.norm(muf,sigf) # Gaussian of forward, v0 scale
            return Transverse_percentage(v)*rv_f.pdf(v)
        
        return integrate.quad(total_func,self.v_trap_initial[0],self.v_trap_initial[-1],limit=100)[0]
        
        
