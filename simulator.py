import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import qutip as q
import scipy.constants as c
from scipy.linalg import expm

hbar = c.hbar
muB = c.value('Bohr magneton')


def simulateSquarePulse(
    psi0,
    b1,
    g, 
    tau, 
    det, 
):
    # Plot full state evolution on Bloch sphere
    NUM_STEPS = 400 
    dt = tau/NUM_STEPS
    U = hardPulsePropagator(b1, g, dt, det)
    psi = psi0
    states = np.zeros((NUM_STEPS, 2 , 1), dtype=complex)

    for i in range(NUM_STEPS):
        states[i,:, :] = psi
        psi = np.matmul(U, psi)
    
    plotBlochSphere(states)
    return 

def hardPulsePropagator(b1, g, dt, det):

    gamma = g*muB/hbar
    # Rotating frame hamiltonian:
    H = np.array([[2*np.pi*det, gamma*b1],
                  [gamma*b1, -2*np.pi*det]]) * 1/2
    
    # Calculate unitary propagator
    U = expm(1.0j*H*dt)

    return U

def plotBlochSphere(states):
    
    points = np.array([state_to_point(s) for s in states]).T

    C1 = '#34A300' # Green
    C2 = '#DD404B' # Red
    gradient = [colorGradient(C1,C2, x/len(states)) for x in range(len(states))]

    b = q.Bloch()
    b.add_points(points)
    b.point_color = gradient
    b.show()

    return b

def predictRabi(b1, g):
    omegaR = 1/2*g*muB*b1/hbar 
    return omegaR/(2*np.pi)


def state_to_point(vector):
    a = complex(vector[0])
    b = complex(vector[1])
    
    phi = np.angle(b) - np.angle(a)
    theta = np.arccos(np.abs(a))*2 # Polar angle

    x = np.sin(theta)*np.cos(phi)
    y = np.sin(theta)*np.sin(phi)
    z = np.cos(theta)

    return [x, y, z]

def colorGradient(c1,c2,r=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1 = np.array(mpl.colors.to_rgb(c1))
    c2 = np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-r)*c1 + r*c2)