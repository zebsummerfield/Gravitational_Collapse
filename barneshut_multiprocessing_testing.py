import numpy as np
from particle import Particle
import distributions
from utils import *
import time
from barneshut import Node
from multiprocessing import Pool

if __name__ == "__main__":

    solar_mass = 2e30
    galaxy_mass = 1e12 * solar_mass
    num = 10000
    mega_particle_mass = (galaxy_mass / num) * 0.5
    galaxy_radius = 5e20
    dt = 10000 * 365 * 24 * 3600
    pc = 3.086e16
    scale_length = 3 * 1000 * pc

    positions = distributions.normal_circle(galaxy_mass, galaxy_radius, num)
    # positions = distributions.disc_exp(galaxy_mass, scale_length, num)
    particles = []
    for pos in positions:
        particles.append(Particle(mega_particle_mass, [pos[0], pos[1], 0], [pos[2], pos[3], 0], dt=dt))

    t = time.time()
    permutate(particles)
    fs = time.time() - t

    t = time.time()
    with Pool() as pool:
        particles = pool.starmap(permutate_v_multi, [(p, particles) for p in particles])
    for p in particles:
        permutate_pos_multi(p)
    fp = time.time() - t

    t = time.time()
    tree = Node(np.zeros(3), 1e21, np.array(particles))
    permutate_tree(tree, particles)
    bs = time.time() - t

    t = time.time()
    tree = Node(np.zeros(3), 1e21, np.array(particles))
    with Pool() as pool:
        particles = pool.starmap(permutate_tree_v_multi, [(p, tree) for p in particles])
    for p in particles:
        permutate_pos_multi(p)
    bp = time.time() - t

    print('Brute Force serial: ' + str(fs))
    print('Brute Force Parallel: ' + str(fp))
    print('Barnes Hut Serial: ' + str(bs))
    print('Barnes Hut Parallel: ' + str(bp))

                