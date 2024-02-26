"""
Animates an NFW halo evolved with a serial Barnes-Hut method.
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

num = 500
mega_particle_mass = (halo_mass / num) * 1
dt = 1000000 * year_in_s
sys.setrecursionlimit(10000)

particles = np.array(distributions.halo_NFW(halo_mass, halo_scale_radius, virial_radius, num, dt=dt))

distances = []
velocities = []
for p in particles:
	distances.append(modulus(p.pos))
	velocities.append(modulus(p.v))
mpl.rcParams["font.size"] = 15
fig, ax = plt.subplots(figsize=(10,10))
ax.plot(np.array(distances)/(pc*1000), np.array(velocities)/1000, linestyle='none', marker='o', markersize=2, label='Velocity Magnitudes')
ax.set_xlabel('Distance from centre [$kpc$]')
ax.set_ylabel('Speed [$kms^{-1}$]')
plt.legend()
plt.show()

fig, axes = plt.subplots(1,2, figsize=(20,10))
positions = np.zeros((len(particles),3))
radii = np.zeros(len(particles))
tree = Node(np.zeros(3), 2*virial_radius, particles, oct=True)
for index, p in enumerate(particles):
	positions[index] = p.pos
	radii[index] = modulus(p.pos)
axes[0].remove()
axes[0] = fig.add_subplot(1,2,1, projection='3d')
plot = axes[0].plot3D(positions[:,0], positions[:,1], positions[:,2], marker='o', linestyle='none', markersize=1)[0]
axes[0].set_xlim(-virial_radius,virial_radius)
axes[0].set_ylim(-virial_radius,virial_radius)
axes[0].set_zlim(-virial_radius,virial_radius)
hist = axes[1].hist(radii, bins=20)
start_energy = 0
for p in particles:
	start_energy += p.calc_kinetic_energy()
	start_energy += p.potential_tree(tree) * 0.5

def update(frame, particles):
	if frame == 1000:
		np.save('particles_1000', particles, allow_pickle=True)
	positions = np.zeros((len(particles),3))
	radii = np.zeros(len(particles))
	tree = Node(np.zeros(3), 2*virial_radius, particles, oct=True)
	if frame%10 == 0:
		KE = 0
		PE = 0
		for p in particles:
			KE += p.calc_kinetic_energy()
			PE += p.potential_tree(tree) * 0.5
		print(f"Frame: {frame}, Kinetic Energy: {KE:.3g}J, Potential Energy: {PE:.3g}J, Fractional Change in Energy: {((KE+PE)/start_energy):.3g}")
	else:
		print(f"Frame: {frame}")
	permutate_tree(tree, particles)
	for index, p in enumerate(particles):
		positions[index] = p.pos
		radii[index] = modulus(p.pos - particles[0].pos)
	plot.set_xdata(positions[:,0])
	plot.set_ydata(positions[:,1])
	plot.set_3d_properties(positions[:,2])
	axes[1].clear()
	axes[1].hist(radii, bins=20)
	return (plot)

ani = animation.FuncAnimation(fig, update,  frames=100000, fargs=([particles]), interval=10)
plt.show()
