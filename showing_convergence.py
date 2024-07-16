"""
Animates a disc galaxy evolved with a serial Barnes-Hut method.
"""

import numpy as np
import distributions
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils import *
import sys
from analytics_total import *
from constants import *
import time
from particle import Particle
from kolmogorov_smirnov import KS_test
import csv

num_disc = 250
num_halo = 500
total_time = 1e8 * year_in_s
sys.setrecursionlimit(10000)

dt = 100000 * year_in_s
centre_particle = np.array([Particle(centre_mass, [0,0,0], [0,0,0], dt=dt)])
disc_particles = distributions.disc_all(disc_mass, scale_radius, halo_scale_radius, centre_mass, disc_density0, halo_density0, num_disc, dt)
halo_particles = distributions.halo_NFW(halo_mass, halo_scale_radius, virial_radius, num_halo, dt)
total_particles = np.concatenate((centre_particle, disc_particles, halo_particles))
np.save('convergence_test_initial_system.npy', total_particles, allow_pickle=True)

dts = [1e6*year_in_s, 3e5*year_in_s, 1e5*year_in_s, 3e4*year_in_s]
KS_test_results = []
z_stds = []
energy_fractions = []
for dt in dts:

	total_particles = np.load('convergence_test_initial_system.npy', allow_pickle=True)
	centre_particle = np.array([total_particles[0]])
	disc_particles = total_particles[1:num_disc+1]
	halo_particles = total_particles[num_disc+1:]

	radii = np.zeros(len(disc_particles))
	for index, p in enumerate(disc_particles):
		radii[index] = modulus(p.pos)
	print(max(KS_test(radii, scale_radius)))

	for p in halo_particles:
		v = np.sqrt(0.5 * abs(p.calc_potential_energy(total_particles)) / p.mass)
		v_dir = np.random.normal(0, 1, 3)
		v_hat = v_dir / np.sqrt(sum(v_dir * v_dir))
		p.v = v * v_hat
	
	for p in disc_particles:
		p.dt = dt
		p.set_circ_v(total_particles)
		p.v += (p.v / modulus(p.v)) * np.random.normal(0, v_dispersion_azimuthal(modulus(p.pos), scale_radius, halo_scale_radius, centre_mass, disc_density0, halo_density0, Q=1))
		p.v += (p.pos / modulus(p.pos)) * np.random.normal(0, v_dispersion_radial(modulus(p.pos), scale_radius, halo_scale_radius, centre_mass, disc_density0, halo_density0, Q=1))

	tree = Node(np.zeros(3), 2*virial_radius, total_particles, oct=True)
	start_KE = 0
	start_PE = 0
	for p in total_particles:
		p.dt = dt
		start_KE += p.calc_kinetic_energy()
		start_PE += p.potential_tree(tree) * 0.5

	for i in range(int(total_time/dt)):
		t = time.time()
		positions = np.zeros((len(disc_particles),3))
		radii = np.zeros(len(disc_particles))
		z = np.zeros(len(disc_particles))
		tree = Node(np.zeros(3), 2*virial_radius, total_particles, oct=True)
		if (i+1)%10 == 0:
			KE = 0
			PE = 0
			for p in total_particles:
				KE += p.calc_kinetic_energy()
				PE += p.potential_tree(tree) * 0.5
			print((f'\nStep = {i},  Time Passed = {round((i+1) * dt/(1000000*year_in_s), 1):.1f} Myrs Fractional Change in Energy = {((KE+PE)/(start_KE+start_PE)):.5g}'))
		permutate_tree(tree, total_particles)
		com = COM(np.concatenate((centre_particle, disc_particles)))
		for index, p in enumerate(disc_particles):
			positions[index] = p.pos
			radii[index] = modulus((p.pos - com)[:2])
			z[index] = ((p.pos-com)[2])
		KS_test_result = KS_test(radii, scale_radius)
		print(f'Step = {i}, Step Time = {(time.time()-t):.5g}, K-S test_statistic = {max(KS_test_result):.5g}, Vertical Standard Deviation = {np.std(z)/pc}')
	
	KS_test_results.append(max(KS_test_result))
	z_stds.append(np.std(z))
	energy_fractions.append((KE+PE)/(start_KE+start_PE))

with open('convergence_KS_tests.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(KS_test_results)

with open('convergence_z_stds.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(z_stds)

with open('convergence_energy.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(energy_fractions)

