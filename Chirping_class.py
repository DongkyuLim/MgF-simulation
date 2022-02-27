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

class Chirping:
    '''
    Useful module that calculate main-laser slowing beam`s motion trace
    
    Parameters
    ----------
    main_detune : float or int
        Frequency detune of main slowing beam
    chirp_coeff : float or int
        Change of frequency detune in 1000000 $t_0$(about 7.2 ms)
    power_rate : float or int
        Power rate between main beam and each sideband.
        For example, power_rate = 1.2 means that main laser`s power is 1.2 times stronger than one sideband`s power.
    laser_on : float or int
        Time when the laser switch is on.
    laser_off : float or int
        Time when the laser switch is off.
    
    '''
    def __init__(self, main_detune, chirp_coeff, power_rate,laser_on, laser_off):
        self.main_detune = main_detune
        self.chirp_coeff = chirp_coeff
        self.power_rate = power_rate
        self.sols = None
        self.v_trap = list()
        self.finals = list()
        self.laser_on = laser_on
        self.laser_off = laser_off

        
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
        

        # The detunings used in the PRAs:
        intensity = 2./(np.pi*(0.020)**2)/Isat


        #Define the hamiltonian
        H0_X, Bq_X, U_X, Xbasis = pylcp.hamiltonians.XFmolecules.Xstate(N=1,I=0.5,B=15496.8125/Gamma,
        gamma = 50.697/Gamma,b=154.7/Gamma, c=178.5/Gamma,gI=5.585,gS=2.0023193043622,
            muB = cts.value('Bohr magneton in Hz/T')/1e6*1e-4/Gamma,
            muN=cts.m_e/cts.m_p*cts.value('Bohr magneton in Hz/T')*1e-4*1e-6/Gamma,return_basis=True
            )

        # b : SI coupling(isotropic), c : Iz Sz coupling(anisotropic), cc : I N coupling, gamma : S N coupling

        E_X = np.unique(np.diag(H0_X))

        H0_A, Bq_A, Abasis = pylcp.hamiltonians.XFmolecules.Astate(J=0.5,I=0.5,
            P=+1,B=15788.2/Gamma,D=0.,H=0.,a=0./Gamma,b=-0.4/Gamma,c=0.,q=0., p=15./Gamma,
            muB=cts.value('Bohr magneton in Hz/T')/1e6*1e-4/Gamma,
            muN=cts.m_e/cts.m_p*cts.value('Bohr magneton in Hz/T')*1e-4*1e-6/Gamma,return_basis=True
            )


        # gJ : Lande g-factor, p : parity(e parity)

        E_A = np.unique(np.diag(H0_A))

        dijq = pylcp.hamiltonians.XFmolecules.dipoleXandAstates(
            Xbasis, Abasis, UX=U_X
            )

        hamiltonian = pylcp.hamiltonian(H0_X, H0_A, Bq_X, Bq_A, dijq,mass = mass)
        
        self.hamiltonian = hamiltonian
        self.magField = np.zeros(3)
        
        
        def Fixed_detune_MgF_MOT(main_det,det_coeff,pr,laseron,laseroff):
            det_side = 120/Gamma
            Avg_X = np.average(E_X)

            init_pow = 0.15*2./(np.pi*(0.020)**2)/Isat
            power_rate = pr/(2+pr)

            def Heav_step(t):
                if laseron<=t and t<laseron+14:
                    return -1*(t-laseron-7)*((t-laseron-7)**2-49*3)*1/686*1/2+1/2
                elif laseron+14<=t and t<laseroff:
                    return 1
                elif t>=laseroff and t<laseroff+14:
                    return (t-laseroff-7)*((t-laseroff-7)**2-49*3)*1/686*1/2 + 1/2
                else:
                    return 0    
                
            def Chirping(t):
                return det_coeff*t*1e-6

            laserBeams = pylcp.laserBeams()
        #Switch
        #Minus sideband part
            laserBeams += pylcp.laserBeams([{'kvec':np.array([0,0,-1]),'pol':+1,'pol_coord':'spherical','s':lambda R,t : Heav_step(t)*init_pow*0.5*(1-power_rate),
                                                           'delta': lambda t : E_A[-1]-Avg_X-main_det-det_side+Chirping(t)}])
            laserBeams += pylcp.laserBeams([{'kvec':np.array([0,0,-1]),'pol':-1,'pol_coord':'spherical','s':lambda R,t : Heav_step(t)*init_pow*0.5*(1-power_rate),
                                                           'delta': lambda t : E_A[-1]-Avg_X-main_det-det_side+Chirping(t)}])

        #Main part
            laserBeams += pylcp.laserBeams([{'kvec':np.array([0,0,-1]),'pol':+1,'pol_coord':'spherical','s':lambda R,t : Heav_step(t)*init_pow*power_rate,
                                                           'delta': lambda t : E_A[-1]-Avg_X-main_det+Chirping(t)}])
            laserBeams += pylcp.laserBeams([{'kvec':np.array([0,0,-1]),'pol':-1,'pol_coord':'spherical','s':lambda R,t : Heav_step(t)*init_pow*power_rate,
                                                           'delta': lambda t : E_A[-1]-Avg_X-main_det+Chirping(t)}])

        #Plus sideband part
            laserBeams += pylcp.laserBeams([{'kvec':np.array([0,0,-1]),'pol':+1,'pol_coord':'spherical','s':lambda R,t : Heav_step(t)*init_pow*0.5*(1-power_rate),
                                                           'delta': lambda t : E_A[-1]-Avg_X-main_det+det_side+Chirping(t)}])
            laserBeams += pylcp.laserBeams([{'kvec':np.array([0,0,-1]),'pol':-1,'pol_coord':'spherical','s':lambda R,t : Heav_step(t)*init_pow*0.5*(1-power_rate),
                                                           'delta': lambda t : E_A[-1]-Avg_X-main_det+det_side+Chirping(t)}])
        # #No-switch
        # #Minus sideband part
        #     laserBeams += pylcp.laserBeams([{'kvec':np.array([0,0,-1]),'pol':+1,'pol_coord':'spherical','s':lambda R,t : init_pow*0.5*(1-power_rate),
        #                                                    'delta': lambda t : E_A[-1]-Avg_X-main_det-det_side+Chirping(t)}])
        #     laserBeams += pylcp.laserBeams([{'kvec':np.array([0,0,-1]),'pol':-1,'pol_coord':'spherical','s':lambda R,t : init_pow*0.5*(1-power_rate),
        #                                                    'delta': lambda t : E_A[-1]-Avg_X-main_det-det_side+Chirping(t)}])

        # #Main part
        #     laserBeams += pylcp.laserBeams([{'kvec':np.array([0,0,-1]),'pol':+1,'pol_coord':'spherical','s':lambda R,t : init_pow*power_rate,
        #                                                    'delta': lambda t : E_A[-1]-Avg_X-main_det+Chirping(t)}])
        #     laserBeams += pylcp.laserBeams([{'kvec':np.array([0,0,-1]),'pol':-1,'pol_coord':'spherical','s':lambda R,t : init_pow*power_rate,
        #                                                    'delta': lambda t : E_A[-1]-Avg_X-main_det+Chirping(t)}])

        # #Plus sideband part
        #     laserBeams += pylcp.laserBeams([{'kvec':np.array([0,0,-1]),'pol':+1,'pol_coord':'spherical','s':lambda R,t : init_pow*0.5*(1-power_rate),
        #                                                    'delta': lambda t : E_A[-1]-Avg_X-main_det+det_side+Chirping(t)}])
        #     laserBeams += pylcp.laserBeams([{'kvec':np.array([0,0,-1]),'pol':-1,'pol_coord':'spherical','s':lambda R,t : init_pow*0.5*(1-power_rate),
        #                                                    'delta': lambda t : E_A[-1]-Avg_X-main_det+det_side+Chirping(t)}])

            return laserBeams
        
        self.laserBeams = Fixed_detune_MgF_MOT(self.main_detune,self.chirp_coeff,self.power_rate,self.laser_on,self.laser_off)
        
        self.rateeq = pylcp.rateeq(self.laserBeams,self.magField,self.hamiltonian,include_mag_forces=0)
        
    def motion_trace(self,vparam:list,tparam:list):
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
        def captured_condition(t, y, threshold=4):
            if y[-4]<4:
                val = -1.
            else:
                val = 1.

            return val
        def lost_condition(t,y):
            if y[-4]>4 and y[-1]>0:
                val = -1
            elif y[-4]<0:
                val = -1
            else:
                val = 1
            return val

        def for_transverse_condition(t,y):
            if y[-1]>0:
                val = -1.
            else:
                val = 1.
            return val

        captured_condition.terminal=False
        lost_condition.terminal=True
        for_transverse_condition.terminal = True
        conditions =  [captured_condition,lost_condition,for_transverse_condition]
        
        v_eval = np.arange(vparam[0],vparam[1],vparam[2])
        t_eval = np.linspace(tparam[0],tparam[1],tparam[2])
        self.v_eval = v_eval
        self.t_eval = t_eval
        
        sols_rate = list()
        for v0a in v_eval:
            self.rateeq.set_initial_position_and_velocity(np.array([0.,0.,-1*self.zz.max()]),
                                                          np.array([0.,0.,v0a]))
            self.rateeq.set_initial_pop(np.array([1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0]))

            self.rateeq.evolve_motion([0.,max(t_eval)],t_eval=t_eval,events= conditions,
                                      progress_bar = 1,method='LSODA')
            sols_rate.append(self.rateeq.sol)
        self.sols = sols_rate
        
        for sol in sols_rate:
            if len(sol.t_events[2])==1:
                self.finals.append(sol.t_events[2][0])
                self.v_trap.append(sol.v[2][0])
        
        
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
        ax.set_xlabel('$z\ (\mathrm{mm})$')
        ax.set_ylabel('$v\ (\mathrm{m/s})$')
        ax.set_title('Z axis motion trace')
        ax.set_xlim(-1*self.zz.max()*self.x0*1000,self.zz.max()*self.x0*1000)
        ax.set_ylim(-1*self.v_eval.max()*self.v0,self.v_eval.max()*self.v0)
        fig.subplots_adjust(left=0.12,right=0.9)

        for sol in self.sols:
            ax.plot(sol.r[2]*self.x0*1000,sol.v[2]*self.v0, 'b')
        ax.plot(self.zz,np.ones(len(self.zz))*30,'r')
        
        if save:
            fig.savefig(save_name)
        
    def percentage(self):
        '''
        Calculate the percent that satisfies both captured_condition and transverse velocity condition,
        with given main_detune, chirp_coeff, power_rate.

        Returns
        -------
        Population : float
            Population that satisfies both captured_condition and transverse velocity condition.

        '''
        if len(self.finals)<=0:
            return 0.
        def t_vs_vf(vff):
            if vff>= min(self.v_trap) and vff<=max(self.v_trap):
                return np.interp(vff,self.v_trap,self.finals)
                
        def vt_vs_vfs(vfs):
            return (12.00/t_vs_vf(vfs)*1e-3/self.t0)
        
        muf = 140/self.v0
        sigf = 17/self.v0
        mut = 0
        sigt = 18.7564/self.v0
        
        def pt_vs_vf(v):
            rv = stats.norm(mut,sigt) # Gaussian of transverse, v0 scale
            val = (rv.cdf(vt_vs_vfs(v))-rv.cdf(0))/rv.cdf(sigt*2)*2
            
            if val >= 1:
                return 1
            else:
                return val
        
        def total_func(v):
            rv_f = stats.norm(muf,sigf) # Gaussian of forward, v0 scale
            return pt_vs_vf(v)*rv_f.pdf(v)
        
        return integrate.quad(total_func,self.v_trap[0],self.v_trap[-1])[0]
        
        
