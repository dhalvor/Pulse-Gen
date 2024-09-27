import numpy as np
import scipy.constants as c
import matplotlib.pyplot as plt
from scipy.linalg import expm
import time

### Set constants used below

hbar = c.hbar
muB = c.value('Bohr magneton')

def mainAnnealingLoop(
    T_max,
    n_max,
    tau,
    Np,
    band_dig,
    pulse_type
):
    E_list = []
    T = T_max # Set initial temperature
    A_coeffs, B_coeffs = initialSquare(n_max, tau, pulse_type) # Get initial params

    # Get initial pulse shape and calc error
    times = np.linspace(0, tau, Np)
    psi0 = np.array([[1],
                     [0]], dtype=complex)
    w1 = coeffs_to_BURP(A_coeffs, B_coeffs, times, tau)
    E = calcError(psi0, w1, n_max, tau, band_dig) # initial error
    E_list.append(E)
    E_diff = 1

    # Uphill move params
    up_attempt_max = (2*n_max + 1)*500
    up_success_max = 0.1*up_attempt_max
    up_attempt = 0
    up_success = 0

    while T > 1e-6 and np.abs(E_diff) > 1e-6:
        # Propose new state
        A_new, B_new = newCoeffs(A_coeffs, B_coeffs)
        w1_new = coeffs_to_BURP(A_new, B_new, times, tau)

        # Accept or reject new state
        E_new = calcError(psi0, w1_new, n_max, tau, band_dig)
        E_diff = E_new - E

        if E_diff <= 0:
            A_coeffs = A_new
            B_coeffs = B_new
            E = E_new
            print('New point accepted')
            E_list.append(E)
        else:
            up_attempt += 1
            R2 = np.random.rand()
            P_acc = np.exp(-E_diff/T)
            if P_acc > R2:
                up_success += 1
                A_coeffs = A_new
                B_coeffs = B_new
                E = E_new
                print('New point accepted')
                E_list.append(E)
            else:
                print('New point rejected')
        
        print(f'New energy: {E}')

        # Update temperature
        if up_attempt == up_attempt_max or up_success == up_success_max:
            T = 0.85*T # 0.85 is the cooling factor (may be changed)
            print(f'T decreased to {T}')
            print(f'Uphill success ratio: {up_success/up_attempt}')
            up_attempt = 0
            up_success = 0

    
    print(f'Loop terminated at E = {E}')

    return A_coeffs, B_coeffs

def initialSquare(n_max, tau, pulse_type):
    A_coeffs = np.zeros(n_max + 1)
    B_coeffs = np.zeros(n_max)
    w = 2*np.pi/tau # Rabi frequency for full rotation in tau
    
    if pulse_type == 'pi':
        A_coeffs[0] = 0.5
    elif pulse_type == 'pi/2':
        A_coeffs[0] = 0.25

    return A_coeffs, B_coeffs


def calcError(
    psi0,
    w1,
    n_max,
    tau,
    band_dig
):  
    pass_det = np.linspace(0, 2/tau, int(2*band_dig))
    stop_det = np.linspace(3/tau, (n_max + 3)/tau, int(n_max*band_dig))
    full_det = np.linspace(0, (n_max + 3)/tau, int((n_max + 3)*band_dig))
    # pass_det = np.linspace(0, 2/tau, int(2*band_dig))
    # stop_det = np.linspace(3/tau, 10/tau, int(7*band_dig))
    # full_det = np.linspace(0, 10/tau, int(10*band_dig))

    ## Following is case for excitation pulse (E-BURP-2)
    # Define ideal profiles
    Py_in_ideal = np.ones(len(pass_det))
    Py_out_ideal = np.zeros(len(stop_det))
    Px_ideal = np.zeros(len(full_det))

    # Calculate profiles for proposed shape
    Py_in = polarisationSpectrum(psi0, w1, tau, pass_det, 'Py')
    Py_out = polarisationSpectrum(psi0, w1, tau, stop_det, 'Py')
    Px = polarisationSpectrum(psi0, w1, tau, full_det, 'Px')

    # plt.plot(pass_det, Py_in_ideal, color='blue', label='Py_ideal')
    # plt.plot(stop_det, Py_out_ideal, color='blue')
    # plt.plot(pass_det, Py_in, color='red', label='Py')
    # plt.plot(stop_det, Py_out, color='red')
    # plt.plot(full_det, Px_ideal, color='green', label='Px_ideal')
    # plt.plot(full_det, Px, color='black', label='Px')
    # plt.legend()

    # Calculate error of proposed w1
    E = np.square(np.subtract(Py_in_ideal, Py_in)).mean() # Fast MSE
    E += np.square(np.subtract(Py_out_ideal, Py_out)).mean()
    E += np.square(np.subtract(Px_ideal, Px)).mean()

    return E/6

def newCoeffs(A_coeffs, B_coeffs):

    R1 = np.random.uniform(-1, 1)
    step_size = 0.1
    A_coeffs[0] = A_coeffs[0] + R1*step_size
    A_coeffs[1] = A_coeffs[1] + R1*step_size
    B_coeffs[0] = B_coeffs[0] + R1*step_size

    for n in range(1, len(B_coeffs)):
        step_size = step_size*0.5
        A_coeffs[n+1] = A_coeffs[n+1] + R1*step_size
        B_coeffs[n] = B_coeffs[n] + R1*step_size

    return A_coeffs, B_coeffs

def coeffs_to_BURP(
    A_coeffs,
    B_coeffs,
    t,
    tau,
):  
    w = 2*np.pi/tau
    w1 = A_coeffs[0]
    for i in range(len(B_coeffs)):
        w1 += A_coeffs[i+1]*np.cos((i+1)*w*t)
        w1 += B_coeffs[i]*np.sin((i+1)*w*t)
    
    return w1 * w

def polarisationSpectrum(
    psi0,
    w1,
    tau,
    det,
    polarisation,
):
    # Calculate final polarisations at each detuning
    spectrum = np.zeros(len(det))
    for d in range(len(det)):
        final_state = evolveState(psi0, w1, tau, det[d])[-1]
        spectrum[d] = calcPolarisations(final_state, polarisation)

    
    return spectrum

def evolveState(psi0, w1, tau, det):
    """
    Evolve initial state under Zeeman hamiltonian
    given some detuning and driving field.
    dt is inferred from length of pulse.
    """
    states = np.zeros((len(w1), 2, 1), dtype=complex)
    dt = tau/len(w1)

    psi = psi0
    for i in range(len(w1)):
        states[i, :, :] = psi
        U = hardPulsePropagator(w1[i], dt, det)
        psi = np.matmul(U, psi)

    return states

def hardPulsePropagator(w1, dt, det):

    # Rotating frame hamiltonian:
    H = np.array([[2*np.pi*det, w1],
                  [w1, -2*np.pi*det]], dtype=complex) * 1/2
    
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
        
    Pi = np.real(np.matmul(np.conjugate(states).T, np.matmul(sigma, states)))

    return Pi[0, 0]

tau = 10e-3
n_max = 8 # max fourier coefficients
Np = 100
times = np.linspace(0, tau, Np)
T_max = 1
band_dig = 4 # 2 samples per interval 1/tau

psi0 = np.array([[1],
                 [0]], dtype=complex)


A_coeffs, B_coeffs = initialSquare(n_max, tau, 'pi/2') # Get initial params 
w1 = coeffs_to_BURP(A_coeffs, B_coeffs, times, tau)

#A_coeffs, B_coeffs = mainAnnealingLoop(T_max, n_max, tau, Np, band_dig, 'pi/2')
st = time.time()
for i in range(0, 10):
    E = calcError(psi0, w1, n_max, tau, band_dig)
    print(E)

et = time.time()
print(f'Elapsed time: {round(et - st, 3)}')