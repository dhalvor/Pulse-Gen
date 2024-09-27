import simulator as sim
import generator as gen
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as c

hbar = c.hbar
muB = c.value('Bohr magneton')

# Global variables:
up = np.array([[1], [0]], dtype=complex)
down = np.array([[0], [1]], dtype=complex)
psi0 = 1/np.sqrt(2) * np.array([[1],
                 [1.0j]], dtype=complex)
psi0 = up


# f_rabi = 100e6
# tau = 1/(4*f_rabi)
# print(tau)
# g = 1.8
# b1 = (2*np.pi*f_rabi)*(2*hbar)/(g*muB)

g = 1.8
tau = 200e-9
sample_rate = 1e9
# bw = 50e6
# b1 = gen.generateSLRPulse(g, tau, sample_rate, bw, 'pi/2', filter_type='pm')

w1 = np.genfromtxt('Xpi2_200.csv', delimiter=',')
gamma = g*muB/hbar
b1 = 2*w1/gamma  # correct for wrong hamiltonain in design algorithm

detunings = np.linspace(-100e6, 100e6, 1000)

sim.simulateBlochSphere(psi0, b1, g, tau, 5e6)
# sim.simulatePolarisations(psi0, b1, g, tau, 5e6)
sim.polarisationSpectrum(psi0, b1, g, tau, detunings)
# sim.projectionSpectrum(psi0, b1, g, tau, detunings, down)
plt.show()
