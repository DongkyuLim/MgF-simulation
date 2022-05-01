import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as cts
import pylcp
import time
import random
from tqdm import tqdm
from scipy.interpolate import interp1d,interp2d
from scipy import stats
from scipy import integrate
from bayes_opt import BayesianOptimization
import json
# import seaborn as sns
from scipy import special 
from scipy import integrate
from multiprocessing import Pool
import pymongo


#Define the constants
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
t0_end=3500000

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
z_max = 0.012/x0
z_start = 0.012/x0
omega = 2*np.pi*(cts.c/wavelength) #Transition frequency
Isat = np.pi*cts.h*cts.c*Gamma*2*np.pi*1e6/3*1/(wavelength)**3
t_eval = np.arange(t0_start,t0_end,t0_step)

# The detunings used in the PRAs:
intensities = 2.*MOT_power*1e-3/(np.pi*0.012**2)/Isat


# Current and coil parameters.
connection = pymongo.MongoClient("mongodb://localhost:27017")
Current = connection.db.Current
max_I = Current.find(limit=1,projection={'_id' : False, 'params' : 1}).sort("target",pymongo.DESCENDING)
I_opt = max_I[0]['params']['I']


def Coil_field(I,R:np.array):
    n = 100
    s = 0.14 # in meter
    rad = 0.14 # in meter
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


from scipy.interpolate import RegularGridInterpolator

def coil_f(xx,yy,zz):
    return Coil_field(I_opt,np.array([xx,yy,zz]))

def data_stream(a, b, c):
    for i, av in enumerate(a):
        for j, bv in enumerate(b):
            for kk, cv in enumerate(c):
                print("Iterating...")
                yield (i, j, kk), (av, bv, cv)

def approx(args):
    return args[0], coil_f(*args[1])

def main():
    start = time.time()

    xs = np.linspace(-0.4,0.4,101)/x0
    ys = np.linspace(-0.4,0.4,101)/x0
    zs = np.linspace(-0.14,0.14,101)/x0

    B = np.zeros((3,101,101,101))

    with Pool(16) as pool:
        result = pool.map(approx, data_stream(xs,ys,zs))
        pool.close()
        pool.join()

    for ii,jj in result:
        B[0][ii] = jj[0]
        B[1][ii] = jj[1]
        B[2][ii] = jj[2]

    np.save("B_3D_interp_" + str(round(I_opt,3)),B)

    print(time.time()-start)

if __name__ == "__main__":
    main()
