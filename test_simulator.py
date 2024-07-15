import simulator as sim
import generator as gen
import numpy as np
import matplotlib.pyplot as plt

# Global variables:
psi0 = np.array([[1],
                 [0]], dtype=complex)

g = 1.8

## Test BURP pulse generator from paper coefficients
tau = 100e-9
bw = 40e6
n = 64 # Number of samples in pulse
det = np.linspace(-60e6, 60e6, 600)
b1 = gen.generateBURPPulse(g, tau, n, bw, 'pi/2')
sim.simulateBlochSphere(psi0, b1, g, tau, 20e6)
sim.polarisationSpectrum(psi0, b1, g, tau, det)
sim.projectionSpectrum(psi0, b1, g, tau, det)
plt.show()

