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

num_disc = 100
num_halo = 199
dt = 100000 * year_in_s
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
	p.v += (p.v / modulus(p.v)) * np.random.normal(0, v_dispersion_azimuthal(modulus(p.pos), scale_radius, halo_scale_radius, centre_mass, disc_density0, halo_density0, Q=0))
	p.v += (p.pos / modulus(p.pos)) * np.random.normal(0, v_dispersion_radial(modulus(p.pos), scale_radius, halo_scale_radius, centre_mass, disc_density0, halo_density0, Q=0))
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

fig, axes = plt.subplots(1, 2, figsize=(20,10))
axes[0].remove()
axes[0] = fig.add_subplot(1,2,1, projection='3d')
plot3D_centre =axes[0].plot3D(centre_particle[0].pos[0], centre_particle[0].pos[1], centre_particle[0].pos[2], marker='o', linestyle='none', markersize=3)[0]

disc_positions = np.zeros((len(disc_particles),3))
radii = np.zeros(len(disc_particles))
for index, p in enumerate(disc_particles):
	disc_positions[index] = p.pos
	radii[index] = modulus(p.pos)
plot3D_disc = axes[0].plot3D(disc_positions[:,0], disc_positions[:,1], disc_positions[:,2], marker='o', linestyle='none', markersize=2)[0]

halo_positions = np.zeros((len(halo_particles),3))
for index, p in enumerate(halo_particles):
	halo_positions[index] = p.pos
plot3D_halo = axes[0].plot3D(halo_positions[:,0], halo_positions[:,1], halo_positions[:,2], marker='o', linestyle='none', markersize=1)[0]

tree = Node(np.zeros(3), 2*virial_radius, total_particles, oct=True)
start_energy = 0
for p in total_particles:
	start_energy += p.calc_kinetic_energy()
	start_energy += p.potential_tree(tree) * 0.5

axes[0].set_xlim(-halo_scale_radius, halo_scale_radius)
axes[0].set_ylim(-halo_scale_radius, halo_scale_radius)
axes[0].set_zlim(-halo_scale_radius, halo_scale_radius)
hist = axes[1].hist(radii, bins=20)

def update(frame, particles):
	t = time.time()
	if frame == 1000:
		np.save('particles_1000', particles[1], allow_pickle=True)

	tree = Node(np.zeros(3), 2*virial_radius, particles[3], oct=True)
	if frame%10 == 0:
		energy = 0
		for p in particles[3]:
			energy += p.calc_kinetic_energy()
			energy += p.potential_tree(tree) * 0.5
		print((f'Frame = {frame},  Time Passed = {round((frame+1) * dt/(100000*year_in_s), 1):.1f} Myrs Fractional Change in Energy = {(energy/start_energy):.g}'))

	permutate_tree(tree, particles[3])
	plot3D_centre.set_xdata(particles[0][0].pos[0])
	plot3D_centre.set_ydata(particles[0][0].pos[1])
	plot3D_centre.set_3d_properties(particles[0][0].pos[2])

	disc_positions = np.zeros((len(particles[1]),3))
	radii = np.zeros(len(particles[1]))
	for index, p in enumerate(particles[1]):
		disc_positions[index] = p.pos
		radii[index] = modulus(p.pos)
	plot3D_disc.set_xdata(disc_positions[:,0])
	plot3D_disc.set_ydata(disc_positions[:,1])
	plot3D_disc.set_3d_properties(disc_positions[:,2])

	halo_positions = np.zeros((len(particles[2]),3))
	for index, p in enumerate(particles[2]):
		halo_positions[index] = p.pos
	plot3D_halo.set_xdata(halo_positions[:,0])
	plot3D_halo.set_ydata(halo_positions[:,1])
	plot3D_halo.set_3d_properties(halo_positions[:,2])

	axes[1].clear()
	axes[1].hist(radii, bins=20)
	print((time.time()-t))
	return (plot3D_centre, plot3D_disc, plot3D_halo)

ani = animation.FuncAnimation(fig, update,  frames=100000, fargs=([[centre_particle, disc_particles, halo_particles, total_particles]]), interval=10)
plt.show()
