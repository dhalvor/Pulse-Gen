import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import qutip as q
import scipy.constants as c
from scipy.linalg import expm

hbar = c.hbar
muB = c.value('Bohr magneton')

def simulatePolarisations( 
    psi0,
    b1,
    g,
    tau,
    det,
    polarisations=['Px','Py','Pz']
):
    if isinstance(b1, (int, float)):
        #Square pulse input
        NUM_STEPS = 400
        b1 = b1*np.ones(NUM_STEPS)

    states = evolveState(psi0, b1, g, tau, det)

    # Calculate polarisations of states
    times = np.linspace(0, tau, len(states))
    Pol = np.zeros((len(polarisations), len(states)))
    
    for p in range(len(polarisations)):
        print(polarisations[p])
        Pol[p, :] = calcPolarisations(states, polarisations[p])
    
    C3 = '#628395' # Blue
    C4 = '#B57E2C' # Brown
    C5 = '#360568' # Purple
    colors = [C3, C4, C5]

    fig = plt.figure(figsize=(7, 4))
    for p in range(len(polarisations)):
        plt.plot(times, Pol[p, :], color=colors[p], label=polarisations[p])

    plt.xlabel('time (s)')
    plt.ylabel(r'Polarisation')
    plt.legend()
    plt.title('State polarisation during pulse.')
    plt.ylim([-1.1, 1.1])

    return

def simulateProjection(
    proj,
    psi0,
    b1,
    g,
    tau,
    det,
):
    if isinstance(b1, (int, float)):
        #Square pulse input
        NUM_STEPS = 400
        b1 = b1*np.ones(NUM_STEPS)

    states = evolveState(psi0, b1, g, tau, det)

    # Project states onto desired state 'proj'
    times = np.linspace(0, tau, len(states))
    P = np.zeros(len(states))
    for s in range(len(states)):
        P[s] = np.abs(np.matmul(np.conjugate(proj).T, states[s]))

    C5 = '#360568' # Purple

    fig = plt.figure(figsize=(7, 4))
    plt.plot(times, P, color=C5)
    plt.xlabel('time (s)')
    plt.ylabel(r'State projection: $|\langle \phi | \psi (t) \rangle|^2$')
    plt.title('State projection during pulse.')
    plt.ylim([-0.1, 1.1])

    return

def simulateBlochSphere(
    psi0,
    b1,
    g, 
    tau, 
    det, 
):
    if isinstance(b1, (int, float)):
        #Square pulse input
        NUM_STEPS = 400
        b1 = b1*np.ones(NUM_STEPS)

    states = evolveState(psi0, b1, g, tau, det)
    
    # Plot states on Bloch sphere
    points = np.array([state_to_point(s) for s in states]).T
    C1 = '#34A300' # Green
    C2 = '#DD404B' # Red
    gradient = [colorGradient(C1,C2, x/len(states)) for x in range(len(states))]

    b = q.Bloch()
    b.add_points(points)
    b.point_color = gradient
    b.show()
    
    return 

def evolveState(psi0, b1, g, tau, det):
    """
    Evolve initial state under Zeeman hamiltonian
    given some detuning and driving field.
    dt is inferred from length of pulse.
    """
    states = np.zeros((len(b1), 2, 1), dtype=complex)
    dt = tau/len(b1)

    psi = psi0
    for i in range(len(b1)):
        states[i, :, :] = psi
        U = hardPulsePropagator(b1[i], g, dt, det)
        psi = np.matmul(U, psi)

    return states

def hardPulsePropagator(b1, g, dt, det):

    gamma = g*muB/hbar
    # Rotating frame hamiltonian:
    H = np.array([[2*np.pi*det, gamma*b1],
                  [gamma*b1, -2*np.pi*det]]) * 1/2
    
    # Calculate unitary propagator
    U = expm(1.0j*H*dt)

    return U

def calcPolarisations(states, Pi):
    if Pi == 'Px':
        sigma = np.array([[0, 1],
                          [1, 0]])
    elif Pi == 'Py':
        sigma = np.array([[0, -1.0j],
                          [1.0j, 0]])
    elif Pi == 'Pz':
        sigma = np.array([[1, 0],
                          [0, -1]])
        
    Pi_array = np.zeros(len(states))    
    for s in range(len(states)):
        Pi_array[s] = np.real(np.matmul(np.conjugate(states[s]).T, np.matmul(sigma, states[s])))

    return Pi_array

def predictRabi(b1, g):
    omegaR = g*muB*b1/hbar 
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