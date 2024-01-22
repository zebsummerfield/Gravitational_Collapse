import numpy as np
from particle import Particle
import distributions
from utils import *
import time
from multiprocessing import Pool
from barneshut import Node

if __name__ == "__main__":

    solar_mass = 2e30
    galaxy_mass = 1e12 * solar_mass
    num = 1000
    mega_particle_mass = (galaxy_mass / num) * 0.5
    galaxy_radius = 5e20
    dt = 10000 * 365 * 24 * 3600
    pc = 3.086e16
    scale_length = 3 * 1000 * pc

    # positions = distributions.normal_circle(galaxy_mass, galaxy_radius, num)
    positions = distributions.disc_exp(galaxy_mass, scale_length, num)
    particles = []
    for pos in positions:
        particles.append(Particle(mega_particle_mass, [pos[0], pos[1], 0], [pos[2], pos[3], 0], dt=dt))

    t = time.time()
    permutate(particles)
    print(time.time() - t)

    t = time.time()
    tree = Node(np.zeros(3), 1e21, particles)
    permutate_tree(tree, particles)
    print(time.time() - t)