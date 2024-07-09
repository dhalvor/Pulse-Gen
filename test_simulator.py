import simulator as sim
import numpy as np
import matplotlib.pyplot as plt

psi0 = np.array([[1],
                 [0]])

b1 = 0.5e-3
g = 1.8
tau = 1/(sim.predictRabi(b1, g))
det = 0

sim.simulateBlochSphere(psi0, b1, g, tau, det)
sim.simulateProjection(psi0, psi0, b1, g, tau, det)
sim.simulatePolarisations(psi0, b1, g, tau, det, polarisations=['Py', 'Pz'])
plt.show()
