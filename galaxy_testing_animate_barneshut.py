"""
Generating a test galaxy of stars.
"""

import numpy as np
import matplotlib.pyplot as plt
from particle import Particle
import distributions
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils import *
import sys

solar_mass = 2e30
galaxy_mass = 1e12 * solar_mass
num = 1000
mega_particle_mass = (galaxy_mass / num) * 0.5
galaxy_radius = 5e20
dt = 10000 * 365 * 24 * 3600
pc = 3.086e16
scale_length = 3 * 1000 * pc
sys.setrecursionlimit(5000)

# positions = distributions.normal_circle(galaxy_mass, galaxy_radius, num)
positions = distributions.disc_exp(galaxy_mass, scale_length, num)
particles = []
for pos in positions:
	particles.append(Particle(mega_particle_mass, [pos[0], pos[1], 0], [pos[2], pos[3], 0], dt=dt))
particles.append(Particle(0.5*galaxy_mass, [0,0,0], [0,0,0], dt=dt))
particles = np.array(particles)

fig, axes = plt.subplots(1, 2, figsize=(20,10))
plot = axes[0].plot(positions[:,0], positions[:,1], marker='o', linestyle='none', markersize=2)[0]
axes[1].set_axis_off()
time_text = axes[1].text(0.35, 0.5, 'Frame 0', fontsize=15)

def update(frame, particles):
	positions = np.zeros((len(particles)+1,3))
	tree = Node(np.zeros(3), 1e21, particles)
	permutate_tree(tree, particles)
	for index, p in enumerate(particles):
		positions[index] = p.pos
	plot.set_xdata(positions[:,0])
	plot.set_ydata(positions[:,1])
	time_text.set_text('Frame ' + str(frame))
	return (plot, time_text)

ani = animation.FuncAnimation(fig, update,  frames=100000, fargs=([particles]), interval=10)
plt.show()



