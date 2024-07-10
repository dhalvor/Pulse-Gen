import simulator as sim
import generator as gen
import numpy as np
import matplotlib.pyplot as plt

psi0 = np.array([[1],
                 [0]], dtype=complex)

g = 1.8
tau = 200e-9 #1/(sim.predictRabi(b1, g))

sample_rate = 1e9 # AWG sample rate
# sample_rate = 250e6 # OPX sample rate
band_width = 100e6
b1 = gen.generateSLRPulse(g, tau, sample_rate, band_width, 'pi/2', filter_type='min')

# sim.simulateBlochSphere(psi0, b1, g, tau, det)
# sim.simulateProjection(psi0, psi0, b1, g, tau, det)
# sim.simulatePolarisations(psi0, b1, g, tau, det, polarisations=['Px','Py', 'Pz'])
det = np.linspace(-200e6, 200e6, 400)
sim.polarisationSpectrum(psi0, b1, g, tau, det)
plt.show()
