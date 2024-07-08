import simulator as sim
import numpy as np
import matplotlib.pyplot as plt

psi0 = np.array([[1],
                 [0]])

b1 = 0.5e-3
g = 1.8
tau = 1/(4*sim.predictRabi(b1, g))
det = 0

sim.simulateSquarePulse(psi0, b1, g, tau, det)
plt.show()
