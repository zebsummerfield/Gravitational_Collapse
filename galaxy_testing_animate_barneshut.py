"""
Animates a disc galaxy evolved with a serial Barnes-Hut method.
"""

import numpy as np
import matplotlib.pyplot as plt
import distributions
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils import *
import sys
from velocities_analytic import *
import matplotlib as mpl
from constants import *
import time

num = 1000
mega_particle_mass = (disc_mass / num) * 1
dt = 100000 * year_in_s
sys.setrecursionlimit(10000)

particles = np.array(distributions.disc_exp(disc_mass, scale_radius, num, dt=dt, mass_c=centre_mass, h=1))

distances = []
velocities = []
dispersions_radial = []
dispersions_azimuthal = []
for p in particles:
	distances.append(modulus(p.pos))
	velocities.append(modulus(p.v))
	dispersions_radial.append(gen_v_dispersion_radial(modulus(p.pos), disc_density0, scale_radius, centre_mass))
	dispersions_azimuthal.append(gen_v_dispersion_azimuthal(modulus(p.pos), disc_density0, scale_radius, centre_mass))
mpl.rcParams["font.size"] = 15
fig, ax = plt.subplots(figsize=(10,10))
ax.plot(np.array(distances)/(pc*1000), np.array(velocities)/1000, linestyle='none', marker='o', markersize=2, label='Velocity Magnitudes')
ax.plot(np.array(distances)/(pc*1000), np.array(dispersions_radial)/1000, linestyle='none', marker='o', markersize=2, label='Radial Dispersions')
ax.plot(np.array(distances)/(pc*1000), np.array(dispersions_azimuthal)/1000, linestyle='none', marker='o', markersize=2, label='Azimuthal Dispersions')
ax.set_xlabel('Distance from centre [$kpc$]')
ax.set_ylabel('Speed [$kms^{-1}$]')
plt.legend()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(24,8))
positions = np.zeros((len(particles),3))
radii = np.zeros(len(particles))
tree = Node(np.zeros(3), 1e21, particles)
start_energy = 0
for index, p in enumerate(particles):
	positions[index] = p.pos
	radii[index] = modulus(p.pos)
	start_energy += p.calc_kinetic_energy()
	start_energy += p.potential_tree(tree) * 0.5
plot = axes[0].plot(positions[:,0], positions[:,1], marker='o', linestyle='none', markersize=1)[0]
axes[0].set_xlim(-3e20,3e20)
axes[0].set_ylim(-3e20,3e20)
axes[1].set_axis_off()
time_text = axes[1].text(0.3, 0.3, 'Time Passed = 0 Myrs', fontsize=15)
energy_text = axes[1].text(0.3, 0.7, f'Fractional Energy = {1}', fontsize=15)
hist = axes[2].hist(radii, bins=20)

def update(frame, particles):
	print(frame)
	t = time.time()
	if frame == 1000:
		np.save('particles_1000', particles, allow_pickle=True)
	positions = np.zeros((len(particles),3))
	radii = np.zeros(len(particles))
	tree = Node(np.zeros(3), 1e21, particles)
	if frame%10 == 0:
		energy = 0
		for p in particles:
			energy += p.calc_kinetic_energy()
			energy += p.potential_tree(tree) * 0.5
		energy_text.set_text(f'Fractional Energy = {(energy/start_energy):.3f}')
	permutate_tree(tree, particles)
	for index, p in enumerate(particles):
		positions[index] = p.pos
		radii[index] = modulus(p.pos - particles[0].pos)
	plot.set_xdata(positions[:,0])
	plot.set_ydata(positions[:,1])
	time_text.set_text(f'Time Passed = {round((frame+1) * dt/(1000000*year_in_s), 1):.1f} Myrs')
	axes[2].clear()
	axes[2].hist(radii, bins=20)
	print((time.time()-t))
	return (plot, time_text, energy_text)

ani = animation.FuncAnimation(fig, update,  frames=100000, fargs=([particles]), interval=10)
plt.show()
