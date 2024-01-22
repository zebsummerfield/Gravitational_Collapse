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
from multiprocessing import Pool

solar_mass = 2e30
galaxy_mass = 1e12 * solar_mass
num = 1000
mega_particle_mass = (galaxy_mass / num) * 0.5
galaxy_radius = 5e20
dt = 10000 * 365 * 24 * 3600
pc = 3.086e16
scale_length = 3 * 1000 * pc

def perm(particles):
	with Pool() as pool:
		print(particles[0].pos)
		particles = pool.starmap(permutate_v_multi, [(p, particles) for p in particles])
		particles = pool.map(permutate_pos_multi, particles)
		print(particles[0].pos)
	# multithreading creates a new particle instance so need to return it
	return particles

def update(frame, particles):
		print(str(particles[0][0].pos) + 'a')
		positions = np.zeros((len(particles[0])+1,3))
		particles[0] = perm(particles[0])
		for index, p in enumerate(particles[0]):
			positions[index] = p.pos
		plot.set_xdata(positions[:,0])
		plot.set_ydata(positions[:,1])
		time_text.set_text('Frame ' + str(frame))
		print(str(positions[0]) + 'b')
		return (plot, time_text)

if __name__ == "__main__":

	# positions = distributions.normal_circle(galaxy_mass, galaxy_radius, num)
	positions = distributions.disc_exp(galaxy_mass, scale_length, num)
	particles = []
	for pos in positions:
		particles.append(Particle(mega_particle_mass, [pos[0], pos[1], 0], [pos[2], pos[3], 0], dt=dt))
	particles.append(Particle(0.5*galaxy_mass, [0,0,0], [0,0,0], dt=dt))

	fig, axes = plt.subplots(1, 2, figsize=(20,10))
	plot = axes[0].plot(positions[:,0], positions[:,1], marker='o', linestyle='none', markersize=2)[0]
	axes[1].set_axis_off()
	time_text = axes[1].text(0.35, 0.5, 'Frame 0', fontsize=15)

	# particles needs to be updated so fargs needs to be a mutable object (list)
	ani = animation.FuncAnimation(fig, update,  frames=100000, fargs=([[particles]]), interval=100)
	plt.show()



