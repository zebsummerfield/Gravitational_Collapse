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
import json

num_disc = 500
num_halo = 1000
dt = 100000 * year_in_s
sys.setrecursionlimit(10000)

centre_particle = np.array([Particle(centre_mass, [0,0,0], [0,0,0], dt=dt)])
disc_particles = distributions.disc_all(disc_mass, scale_radius, halo_scale_radius, centre_mass, disc_density0, halo_density0, num_disc, dt)
halo_particles = distributions.halo_NFW(halo_mass, halo_scale_radius, virial_radius, num_halo, dt)
total_particles = np.concatenate((centre_particle, disc_particles, halo_particles))
np.save('final_stats_start_system.npy', total_particles, allow_pickle=True)

for q in [15]:

	total_particles = np.load('final_stats_start_system.npy', allow_pickle=True)
	centre_particle = np.array([total_particles[0]])
	disc_particles = total_particles[1:num_disc+1]
	halo_particles = total_particles[num_disc+1:]

	for p in halo_particles:
		v = np.sqrt(0.5 * abs(p.calc_potential_energy(total_particles)) / p.mass)
		v_dir = np.random.normal(0, 1, 3)
		v_hat = v_dir / np.sqrt(sum(v_dir * v_dir))
		p.v = v * v_hat

	for p in disc_particles:
		p.set_circ_v(total_particles)
		p.v += (p.v / modulus(p.v)) * np.random.normal(0, v_dispersion_azimuthal(modulus(p.pos), scale_radius, halo_scale_radius, centre_mass, disc_density0, halo_density0, Q=q/10))
		p.v += (p.pos / modulus(p.pos)) * np.random.normal(0, v_dispersion_radial(modulus(p.pos), scale_radius, halo_scale_radius, centre_mass, disc_density0, halo_density0, Q=q/10))

	tree = Node(np.zeros(3), 2*virial_radius, total_particles, oct=True)
	start_KE = 0
	start_PE = 0
	for p in total_particles:
		start_KE += p.calc_kinetic_energy()
		start_PE += p.potential_tree(tree) * 0.5
	radii = []
	positions = []
	com = COM(np.concatenate((centre_particle, disc_particles)))
	for p in disc_particles:
		radii.append(modulus(p.pos - com))
		positions.append(list(p.pos))

	stats_dict = {'time_step': dt, 'num_disc': num_disc, 'num_halo': num_halo, 'start_radii': list(np.sort(radii)), 'start_pos': positions}
	evolving_KE = [start_KE]
	evolving_PE = [start_PE]
	evolving_KS = [max(KS_test(radii, scale_radius))]
	z_std = [0]

	for i in range(10000):
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
			evolving_KE.append(KE)
			evolving_PE.append(PE)
			print((f'\nStep = {i},  Time Passed = {round((i+1) * dt/(1000000*year_in_s), 1):.1f} Myrs Fractional Change in Energy = {((KE+PE)/(start_KE+start_PE)):.5g}'))
		permutate_tree(tree, total_particles)
		com = COM(np.concatenate((centre_particle, disc_particles)))
		for index, p in enumerate(disc_particles):
			positions[index] = p.pos
			radii[index] = modulus((p.pos - com)[:2])
			z[index] = ((p.pos-com)[2])
		KS_test_result = KS_test(radii, scale_radius)
		print(f'Step = {i}, Step Time = {(time.time()-t):.5g}, K-S test_statistic = {max(KS_test_result):.5g}')
		evolving_KS.append(max(KS_test_result))
		z_std.append(np.std(z))
		if i > 500 and max(KS_test_result) < 0.06:
			break

	stats_dict['evolving_KE'] = evolving_KE
	stats_dict['evolving_PE'] = evolving_PE
	stats_dict['evolving_KS'] = evolving_KS
	stats_dict['final_radii'] = list(np.sort(radii))
	stats_dict['final_KS'] = KS_test_result
	stats_dict['z_std'] = z_std
	positions = []
	for p in disc_particles:
		positions.append(list(p.pos))
	stats_dict['final_pos'] = positions

	with open('system_q'+str(q), 'w') as f:
		json.dump(stats_dict, f)
