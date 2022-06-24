import time, os
import pylcp
import numpy as np
import scipy.constants as cts
import matplotlib.pyplot as plt
import time
import pymongo

connection = pymongo.MongoClient("mongodb://localhost:27017")
vc = connection.db.New_vc
max_parameters = vc.find(projection={"_id":0},limit=5).sort("target",pymongo.DESCENDING)[0]["params"]


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
t0_end=7000000

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
mag_field_grad = 1252.8168984164048*x0
waist = 0.012/x0
v_max = 8.
z_max = 0.012/x0
z_start = 0.012/x0
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

def Fixed_detune_MgF_MOT(d1,d2,d3,d4,p1,p2,p3,p4,pol1,pol2,pol3,pol4):
    pX = (p1+p2+p3+p4)/4
    pp1 = p1 - pX
    pp2 = p2 - pX
    pp3 = p3- pX
    pp4 = p4 - pX
    
    def intensity_res(pp):
        intensity_norm = 0.050
        result = 2.*(intensity_norm+pp/(10**3))/(np.pi*(0.012)**2)/Isat
        return result
    
    def sigmoid_function(x):
        if (1/(1+np.exp(-1*x))*2)//1 == 1.:
            return +1
        else:
            return -11
    
    laserBeams = pylcp.laserBeams()
    for ii, Eg_i in enumerate(E_X):
        if ii<1:
            laserBeams += pylcp.conventional3DMOTBeams(
                s=intensity_res(pp1), delta=(E_A[-1] - Eg_i)+d1,pol=sigmoid_function(pol1),wb=waist,
                beam_type=pylcp.gaussianBeam
                )
        elif ii==1:
            laserBeams += pylcp.conventional3DMOTBeams(
                s=intensity_res(pp2), delta=(E_A[-1] - Eg_i)+d2,pol=sigmoid_function(pol2),wb=waist,
                beam_type=pylcp.gaussianBeam)
        elif ii==2:
            laserBeams += pylcp.conventional3DMOTBeams(
                s=intensity_res(pp3), delta=(E_A[-1] - Eg_i)+d3,pol=sigmoid_function(pol3),wb=waist,
                beam_type=pylcp.gaussianBeam
                )
        else:
            laserBeams += pylcp.conventional3DMOTBeams(
                s=intensity_res(pp4), delta=(E_A[-1] - Eg_i)+d4,pol=sigmoid_function(pol4),wb=waist,
                beam_type=pylcp.gaussianBeam
                )

    return laserBeams




def trans_position(z,v0s,Bgrad, d1,d2,d3,d4,p1,p2,p3,p4,pol1,pol2,pol3,pol4):
    v_list = np.linspace(v0s-2,v0s+2,11)
    zz = np.linspace(-1*z_max,z_max,100)
    vv = np.linspace(-1*v_max,v_max,100)
    Z,V = np.meshgrid(zz,vv)
    sols_list=list()
    
    laserBeams = Fixed_detune_MgF_MOT(d1,d2,d3,d4,p1,p2,p3,p4,pol1,pol2,pol3,pol4)
    magField=pylcp.quadrupoleMagneticField(Bgrad*x0*1000)
    eqn = pylcp.rateeq(laserBeams=laserBeams,magField=magField,hamitlonian=hamiltonian)

    def trap_condition(t,y):
        if abs(y[-3])*1000*x0<6 and abs(y[-6])<5e-2 and abs(y[-2])*1000*x0<6 and abs(y[-5])<5e-2 and abs(y[1])*x0*1000<6 and abs(y[-4])<5e-2 and y[-6]*y[-4]<0 and y[-5]*y[-2]<0:
            val = -1.
        else:
            val = 1.
        return val
    
    def lost_condition(t,y):
        if y[-3]*1000*x0>13 or y[-2]*1000*x0>13:
            val = -1.
        else:
            val=1.
        return val
    trap_condition.terminal = True
    lost_condition.terminal = True
    
    for v0ss in v_list:
        eqn.set_initial_position_and_velocity(np.array([-1*z_start,-1*z_start,0]),np.array([v0ss,v0ss,0]))
        eqn.set_initial_pop(np.array([1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0]))

        eqn.evolve_motion([0.,max(t_eval)],t_eval=t_eval,events= [trap_condition,lost_condition],progress_bar = 1,method='LSODA')
        sols_list.append(eqn.sol)
    return sols_list
    
def test(Z):
    return trans_position(Z,**max_parameters)


def if_trapped(z,vz,vc,v0s,Bgrad,d1,d2,d3,d4,p1,p2,p3,p4,pol1,pol2,pol3,pol4):
    def trap_condition(t,y):
        if abs(y[-3])*1000*x0<6 and abs(y[-6])<5e-2 and abs(y[-2])*1000*x0<6 and abs(y[-5])<5e-2 and y[-3]*y[-6] <= 0 and y[-5]*y[-2]<=0:
            val = -1.
        else:
            val = 1.
        return val
    
    def lost_condition(t,y):
        if y[-3]*1000*x0>12 or y[-2]*1000*x0>12:
            val = -1.
        else:
            val=1.
        return val
    trap_condition.terminal = True
    lost_condition.terminal = True
    
    laserBeams = Fixed_detune_MgF_MOT(d1,d2,d3,d4,p1,p2,p3,p4,pol1,pol2,pol3,pol4)
    magField = pylcp.quadrupoleMagneticField(Bgrad*x0*1000)
    eqn = pylcp.rateeq(laserBeams=laserBeams,magField=magField,hamitlonian=hamiltonian)
    
    eqn.set_initial_position_and_velocity(np.array([-1*z_start,-1*z_start,z/x0*1e-3]),np.array([vc,vc,vz]))
    eqn.set_initial_pop(np.array([1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0]))

    eqn.evolve_motion([0.,max(t_eval)],t_eval=t_eval,events= [trap_condition,lost_condition],progress_bar = 0,method='LSODA',max_step=2e5)
    sol = eqn.sol

    if len(sol.t_events[0])==1:
        return 1
    elif len(sol.t_events[1])==1:
        return -1
    else:
        return 0.

def approx(args):
    return args[0], test2(*args[1])

def test2(z,vz,vc):
    return if_trapped(z,vz,vc,**max_parameters)

if __name__ =="__main__":
    print(max_parameters)
    sols = test(0)

    fig,ax = plt.subplots(1,1)


    ax.set_xlabel('$x (\mathrm{mm})$')
    ax.set_ylabel('$v (\mathrm{m/s})$')
    ax.set_title('X axis force image')
    ax.set_xlim(-1*z_max*x0*1000,z_max*x0*1000)
    ax.set_ylim(-1*v_max*v0,1*v_max*v0)

    for sol in sols:
        if len(sol.t_events[0]) == 1:
            print("Trapped")
        ax.plot(sol.r[0]*1000*x0,sol.v[0]*v0)
    plt.show(block=True)