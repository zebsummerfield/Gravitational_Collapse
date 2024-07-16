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
from analytics_total import *
import matplotlib as mpl
from constants import *
import time
from particle import Particle
from kolmogorov_smirnov import KS_test

num_disc = 500
num_halo = 1000
dt = 1000000 * year_in_s
sys.setrecursionlimit(10000)

centre_particle = np.array([Particle(centre_mass, [0,0,0], [0,0,0], dt=dt)])
disc_particles = distributions.disc_all(disc_mass, scale_radius, halo_scale_radius, centre_mass, disc_density0, halo_density0, num_disc, dt)
halo_particles = distributions.halo_NFW(halo_mass, halo_scale_radius, virial_radius, num_halo, dt)
total_particles = np.concatenate((centre_particle, disc_particles, halo_particles))

for p in halo_particles:
	v = np.sqrt(0.5 * abs(p.calc_potential_energy(total_particles)) / p.mass)
	v_dir = np.random.normal(0, 1, 3)
	v_hat = v_dir / np.sqrt(sum(v_dir * v_dir))
	p.v = v * v_hat

distances = []
velocities_analytic = []
velocities_numeric = []
dispersions_radial = []
dispersions_azimuthal = []
velocities_post_dispersion = []
for p in disc_particles:
	distances.append(modulus(p.pos))
	velocities_analytic.append(modulus(p.v))
	dispersions_radial.append(v_dispersion_radial(modulus(p.pos), scale_radius, halo_scale_radius, centre_mass, disc_density0, halo_density0))
	dispersions_azimuthal.append(v_dispersion_azimuthal(modulus(p.pos), scale_radius, halo_scale_radius, centre_mass, disc_density0, halo_density0))
	p.set_circ_v(total_particles)
	velocities_numeric. append(modulus(p.v))
	p.v += (p.v / modulus(p.v)) * np.random.normal(0, v_dispersion_azimuthal(modulus(p.pos), scale_radius, halo_scale_radius, centre_mass, disc_density0, halo_density0, Q=4))
	p.v += (p.pos / modulus(p.pos)) * np.random.normal(0, v_dispersion_radial(modulus(p.pos), scale_radius, halo_scale_radius, centre_mass, disc_density0, halo_density0, Q=4))
	velocities_post_dispersion. append(modulus(p.v))
mpl.rcParams["font.size"] = 15
fig, ax = plt.subplots(figsize=(10,10))
#ax.plot(np.array(distances)/(pc*1000), np.array(velocities_analytic)/1000, linestyle='none', marker='o', markersize=2, label='Analytic Velocity Magnitudes')
#ax.plot(np.array(distances)/(pc*1000), np.array(velocities_numeric)/1000, linestyle='none', marker='o', markersize=2, label='Numeric Velocity Magnitudes')
ax.plot(np.array(distances)/(pc*1000), np.array(dispersions_radial)/1000, linestyle='none', marker='o', markersize=2, label='Radial Dispersions')
ax.plot(np.array(distances)/(pc*1000), np.array(dispersions_azimuthal)/1000, linestyle='none', marker='o', markersize=2, label='Azimuthal Dispersions')
ax.plot(np.array(distances)/(pc*1000), np.array(velocities_post_dispersion)/1000, linestyle='none', marker='o', markersize=2, label='Velocity Magnitudes After Dispersions')
ax.set_xlabel('Distance from centre [$kpc$]')
ax.set_ylabel('Speed [$kms^{-1}$]')
plt.legend()
plt.show()

#total_particles = np.load('convergence_test_initial_system.npy', allow_pickle=True)
centre_particle = np.array([total_particles[0]])
disc_particles = total_particles[1:num_disc+1]
halo_particles = total_particles[num_disc+1:]

fig, axes = plt.subplots(1, 2, figsize=(20,10))
positions = np.zeros((len(disc_particles),3))
radii = np.zeros(len(disc_particles))
for index, p in enumerate(disc_particles):
	positions[index] = p.pos
	radii[index] = modulus(p.pos)

tree = Node(np.zeros(3), 2*virial_radius, total_particles, oct=True)
start_energy = 0
for p in total_particles:
	start_energy += p.calc_kinetic_energy()
	start_energy += p.potential_tree(tree) * 0.5

plot = axes[0].plot(positions[:,0], positions[:,1], marker='o', linestyle='none', markersize=1)[0]
axes[0].set_xlim(-5e20,5e20)
axes[0].set_ylim(-5e20,5e20)
hist = axes[1].hist(radii, bins=20)

def update(frame, particles):
	t = time.time()
	if frame == 1000:
		np.save('particles_1000', particles[1], allow_pickle=True)
	positions = np.zeros((len(particles[1]),3))
	radii = np.zeros(len(particles[1]))
	z = np.zeros(len(particles[1]))
	tree = Node(np.zeros(3), 2*virial_radius, particles[3], oct=True)
	if frame%10 == 0:
		energy = 0
		for p in particles[3]:
			energy += p.calc_kinetic_energy()
			energy += p.potential_tree(tree) * 0.5
		print((f'\nFrame = {frame},  Time Passed = {round((frame+1) * dt/(1000000*year_in_s), 1):.1f} Myrs Fractional Change in Energy = {(energy/start_energy):.5g}'))
	permutate_tree(tree, particles[3])
	com = COM(np.concatenate((particles[0], particles[1])))
	for index, p in enumerate(particles[1]):
		positions[index] = p.pos
		radii[index] = modulus((p.pos - com)[:2])
		z[index] = ((p.pos-com)[2])
	plot.set_xdata(positions[:,0])
	plot.set_ydata(positions[:,1])
	axes[1].clear()
	axes[1].hist(radii, bins=20)
	print(f'Frame = {frame}, Permutation Time = {(time.time()-t):.5g}, K-S test_statistic = {max(KS_test(radii, scale_radius)):.5g}, Vertical Standard Deviation = {np.std(z)/pc}')
	return (plot)

ani = animation.FuncAnimation(fig, update,  frames=100000, fargs=([[centre_particle, disc_particles, halo_particles, total_particles]]), interval=10)
plt.show()
