"""
Loads and shows the state of a system of particles after 1000 permutations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from utils import *
from velocities_analytic import *
import matplotlib as mpl

particles = np.load('particles_1000.npy', allow_pickle=True)

fig, axes = plt.subplots(1, 3, figsize=(24,8))
positions = np.zeros((len(particles),3))
radii = np.zeros(len(particles))
tree = Node(np.zeros(3), 1e21, particles)
start_energy = 0
for index, p in enumerate(particles):
	positions[index] = p.pos
	radii[index] = modulus(p.pos)
	start_energy += p.calc_total_energy(tree)
plot = axes[0].plot(positions[:,0], positions[:,1], marker='o', linestyle='none', markersize=1)[0]
# axes[0].set_xlim(-3e20,3e20)
# axes[0].set_ylim(-3e20,3e20)
axes[1].set_axis_off()
time_text = axes[1].text(0.3, 0.5, f'Time Passed = {1000*0.2} Myrs', fontsize=15)
hist = axes[2].hist(radii, bins=20)

plt.show()

