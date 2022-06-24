import multiprocessing
from multiprocessing.sharedctypes import Value
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as cts
import pylcp
import time
import random
from sympy import root
from tqdm import tqdm
from scipy.interpolate import interp1d,interp2d,RegularGridInterpolator
from scipy import stats
from scipy import integrate
from scipy.optimize import root_scalar
from celluloid import Camera
from bayes_opt import BayesianOptimization
import json
import seaborn as sns
import White_class
from scipy import special
import pymongo
import parmap



#Main variables

main_detune = 17
sideband_detune = 120
white_detune = 10
laser_on = 200000
laser_off = 400015
MOT_power = 50
v0_start=1
v0_step=1
v0_end=25
t0_start=0
t0_step=1
t0_end=10000000


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
waist = 0.012/x0
v_max = 8.
z_max = 384.855e-3/x0
z_start = 384.855e-3/x0
omega = 2*np.pi*(cts.c/wavelength) #Transition frequency
Isat = np.pi*cts.h*cts.c*Gamma*2*np.pi*1e6/3*1/(wavelength)**3
t_eval = np.arange(t0_start,t0_end,t0_step)

# The detunings used in the PRAs:
intensities = 2.*MOT_power*1e-3/(np.pi*0.012**2)/Isat

#Define the hamiltonian
H0_X, Bq_X, U_X, Xbasis = pylcp.hamiltonians.XFmolecules.Xstate(N=1,I=0.5,B=15496.8125/Gamma,
gamma = 50.697/Gamma,b=154.7/Gamma, c=178.5/Gamma,gI=5.585,gS=2.0023193043622,
    muB = cts.value('Bohr magneton in Hz/T')/1e6*1e-4/Gamma,
    muN=cts.m_e/cts.m_p*cts.value('Bohr magneton in Hz/T')*1e-4*1e-6/Gamma,return_basis=True
    )

# b : SI coupling(isotropic), c : Iz Sz coupling(anisotropic), cc : I N coupling, gamma : S N coupling

E_X = np.unique(np.diag(H0_X))

H0_A, Bq_A, Abasis = pylcp.hamiltonians.XFmolecules.Astate(J=0.5,I=0.5,
    P=+1,B=15788.2/Gamma,D=0.,H=0.,a=109./Gamma,b=-299.2/Gamma,c=274.2/Gamma,q=0., p=15./Gamma,
    muB=cts.value('Bohr magneton in Hz/T')/1e6*1e-4/Gamma,
    muN=cts.m_e/cts.m_p*cts.value('Bohr magneton in Hz/T')*1e-4*1e-6/Gamma,
    gl=53/(2*15788.2),glprime=15/(2*15788.2),greprime=0.,return_basis=True
    )
# gJ : Lande g-factor, p : parity(e parity)

E_A = np.unique(np.diag(H0_A))

dijq = pylcp.hamiltonians.XFmolecules.dipoleXandAstates(
    Xbasis, Abasis, UX=U_X
    )

hamiltonian = pylcp.hamiltonian(H0_X, H0_A, Bq_X, Bq_A, dijq,mass = mass)


def Fixed_detune_MgF_MOT(main_det,det_1,det_2,beta_1,beta_2,laseron,laseroff):
    det_side = det_1/Gamma
    det_side2 = det_2/Gamma
    Avg_X = np.average(E_X)
    init_pow = 0.5*2./(np.pi*(0.012)**2)/Isat
    
    def Gaussian_Beam_Intensity(R,waist):
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
        
    
    def pick_EOM(b):
        N_list = range(round(-b)-2,round(b)+2)
        order_list = list()
        # intensity_list = list()
        for n in N_list:
            temp = Bessel_Intensity(n,b)
            if temp>=0.01:
                order_list.append(n)
                # intensity_list.append(temp)``
    
        # return order_list, intensity_list
        return order_list
    
    def laser_set(m,n):
        return pylcp.laserBeams([{'kvec':np.array([-1,-1,0]),'pol':+1,'pol_coord':'spherical','delta':(E_A[-1]-Avg_X-main_det)+m*det_side+det_side2*n,
                                     's': lambda R,t : init_pow*Gaussian_Beam_Intensity(R,waist)*Heav_step(t)*Bessel_Intensity(m,beta_1)*Bessel_Intensity(n,beta_2)},
                                    {'kvec':np.array([-1,-1,0]),'pol':-1,'pol_coord':'spherical','delta':(E_A[-1]-Avg_X-main_det)+m*det_side-det_side2*n,
                                     's': lambda R,t : init_pow*Gaussian_Beam_Intensity(R,waist)*Heav_step(t)*Bessel_Intensity(m,beta_1)*Bessel_Intensity(n,beta_2)}])
    
    
    white_order = pick_EOM(beta_2)
    
    laserBeams = pylcp.laserBeams()
    for m in {-1,0,1}:
        for n in white_order:
            laserBeams+=laser_set(m,n)

    return laserBeams


xs = np.linspace(-0.4,0.4,101)/x0
ys = np.linspace(-0.4,0.4,101)/x0
zs = np.linspace(-0.2,0.2,101)/x0

X,Y,Z = np.meshgrid(xs,ys,zs,sparse=1,indexing="ij")
B = np.load("D:/migration/B_3D_interp_41.21.npy")

Bx = RegularGridInterpolator((xs,ys,zs),B[0])
By = RegularGridInterpolator((xs,ys,zs),B[1])
Bz = RegularGridInterpolator((xs,ys,zs),B[2])

def B_func(R:np.array):
    if abs(R[2])>0.2/x0 or abs(R[1])>0.4/x0 or abs(R[0])>0.4/x0:
        return np.zeros(3,)
    return np.array([Bx(R),By(R),Bz(R)]).reshape(-1)


def motion_trace(v0_l,main_det,det_1,det_2,beta_1,beta_2,laseron,laseroff):
    magField = lambda R,t : B_func(R)
    laserBeams = Fixed_detune_MgF_MOT(main_det,det_1,det_2,beta_1,beta_2,laseron,laseroff)
    rateeq = pylcp.rateeq(laserBeams=laserBeams,magField=magField,hamitlonian=hamiltonian)

    
    def Lost_condition(t,y,threshold = 0.):
        if y[-6]<threshold:
            val = -1.
        else:
            val = 1.
        return val

    def for_transverse_condition(t,y,threshold = -0.012/x0):
        if y[-3]>threshold and y[-2]>threshold:
            val = -1.
        else:
            val = 1.
        return val

    # Capture_velocity_condition.terminal = True
    Lost_condition.terminal = True
    for_transverse_condition.terminal = True
    # conditions =  [for_transverse_condition,Lost_condition,Capture_velocity_condition]
    conditions =  [for_transverse_condition,Lost_condition]    
    v_longitudinal = np.linspace(14,21,16)
    time_final = list()
    v_trap_initial = list()
    
    rateeq.set_initial_position_and_velocity(np.array([-1*z_start/np.sqrt(2),-1*z_start/np.sqrt(2),0]),np.array([v0_l/np.sqrt(2),v0_l/np.sqrt(2),0]))
    rateeq.set_initial_pop(np.array([1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0]))

    rateeq.evolve_motion([0.,max(t_eval)],t_eval=t_eval,events= conditions,max_step=2e5,progress_bar = 0,method='LSODA')
    return rateeq.sol


def multi_motion_trace(main_det,det_1,det_2,beta_1,beta_2,laseron,laseroff):
    v0_longitudinal = np.linspace(14,21,8)

    result = parmap.map(motion_trace,v0_longitudinal,main_det,det_1,det_2,beta_1,beta_2,laseron,laseroff,pm_pbar=0,pm_processes=multiprocessing.cpu_count()-1)
    return result


def tv_generator(sols):
    tv_list = list()
    v_initial = list()
    for sol in sols:
        if len(sol.t_events[0])==1:
            if sol.v[0][-1]<7.:
                tv_list.append((sol.t[-1],sol.v[0][-1]))
                v_initial.append(sol.v[0][0])
    return tv_list, v_initial

z_range = np.linspace(0,12,13) # Unit : mm
vz_range = np.linspace(0,3,13) # Unit : Gamma/k
vc_range = np.linspace(1,7,13) # Unit : Gamma/k

Result_array = np.load("D:/migration/VC_finding_array_0624.npy")

trap_lost = RegularGridInterpolator((z_range,vz_range,vc_range),Result_array[0])

def capture_condition(z,vz,vc):
    try:
        if vc<1 and vc>0:
            result = trap_lost([z,vz,1])[0]
        else:
            result = trap_lost([z,vz,vc])[0]
    except(ValueError):
        result = -1.
    finally:
        return result


def root_find(vz,tt,vc):
    return capture_condition(vz*tt*x0*1000,vz,vc)

def max_vz_calculator(tt,vc):
    try:
        max_vz = root_scalar(root_find,(tt,vc),method='bisect',bracket=[0,1])
    except(ValueError):
        return 0.
    
    return max_vz.root

def multi_max_vz_calculator(tv_list):
    max_vz_list = parmap.starmap(max_vz_calculator,tv_list,pm_pbar=0,pm_processes=15)
    return max_vz_list

def fraction_generator(v_initial, max_vz_list):
    if len(v_initial)<=1:
        return 0
    m_trans = 0.
    std_trans = 18.7564/v0
    m_long = 140/v0
    std_long = 17/v0

    dist_trans = stats.norm(m_trans,std_trans)
    dist_long = stats.norm(m_long,std_long)

    trans_vs_long = interp1d(v_initial,max_vz_list)

    def trans_fraction(v_long):
        result = (dist_trans.cdf(trans_vs_long(v_long))-dist_trans.cdf(0))/(dist_trans.cdf(2*std_trans)-dist_trans.cdf(-2*std_trans))*2
        if result>1:
            return 1
        else:
            return result
    
    def total_fraction(v_long):
        return dist_long.pdf(v_long)*trans_fraction(v_long)

    return integrate.quad(total_fraction,v_initial[0],v_initial[-1],limit=100)[0]

def Expected_simulation(main_det,det_1,det_2,beta_1,beta_2,laseron,laseroff):
    sols = multi_motion_trace(main_det,det_1,det_2,beta_1,beta_2,laseron,laseroff)
    tv_list, v_initial = tv_generator(sols)
    max_vz_list = multi_max_vz_calculator(tv_list=tv_list)
    result = fraction_generator(v_initial=v_initial,max_vz_list=max_vz_list)
    return result


if __name__ == "__main__":
    connection = pymongo.MongoClient("localhost:27017")
    db = connection.db.Expected_model
    max_parameters = db.find(projection={"_id":0},limit=5).sort("target",pymongo.DESCENDING)  
    result = Expected_simulation(**max_parameters[0]["params"])
    print(result)