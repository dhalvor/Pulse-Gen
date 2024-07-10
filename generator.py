import numpy as np
import scipy.constants as c
import matplotlib.pyplot as plt
from scipy.linalg import expm
from sigpy.mri.rf import slr

### Set constants used below

hbar = c.hbar
muB = c.value('Bohr magneton')


### Pulse generating functions
def generateSLRPulse(
    g,
    tau,
    sr,
    bw,
    pulse_type,
    filter_type='min',
    two_axis_pulse=False,
    plot_output=True
):
    """
    Uses Shinnar Le-Roux algorithm to design amplitude modulated pulse.
    Args:
        - tau: pulse time length
        - sr: sample rate of pule generator
            - 1 GHz for AWG
            - 250 MHz for OPX ?
        - bw: desired bandwidth in polarisation profile
    """
    n = int(tau*sr) 
    dt = tau/n
    TBW = tau*bw
    gamma = g*muB/hbar
    scale = 1/(gamma*dt)

    if pulse_type == 'pi':
        ptype = 'inv'
    elif pulse_type == 'pi/2':
        ptype = 'ex'
    else:
        print('Pulse type not possible by SLR. Set either pi or pi/2')

    rf = scale * slr.dzrf(n, TBW, ptype=ptype, ftype=filter_type, d1=0.00001, d2=0.00001, cancel_alpha_phs= not two_axis_pulse)
    if two_axis_pulse == True:
        b1x = np.real(rf)
        b1y = np.imag(rf)
    elif two_axis_pulse == False:
        b1 = np.real(rf)

    if plot_output == True:
        times = np.linspace(0, tau, n)
        C5 = '#360568' # Purple
        fig = plt.figure(figsize=(7, 4))
        plt.plot(times, b1, color=C5)
        plt.xlabel('time (s)')
        plt.ylabel('B1 amplitude (T)')
        plt.title(f'SLR {pulse_type} pulse')

    return b1