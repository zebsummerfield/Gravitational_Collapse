"""
Benchmarks the 4 methods of particle system evolution for different numbers of particles and saves the times to a json.
"""


import numpy as np
from particle import Particle
import distributions
from utils import *
import time
from barneshut import Node
from multiprocessing import Pool
import matplotlib.pyplot as plt
import matplotlib as mpl
import json

if __name__ == "__main__":

    nums = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    fsl = []
    fpl = []
    bsl = []
    bpl = []

    for num in nums:
        fsl_temp = []
        fpl_temp = []
        bsl_temp = []
        bpl_temp = []
        for i in range(10):

            solar_mass = 2e30
            galaxy_mass = 1e12 * solar_mass
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
            fsl_temp.append(fs)

            t = time.time()
            with Pool() as pool:
                particles = pool.starmap(permutate_v_multi, [(p, particles) for p in particles])
            for p in particles:
                permutate_pos_multi(p)
            fp = time.time() - t
            fpl_temp.append(fp)

            t = time.time()
            tree = Node(np.zeros(3), 1e21, np.array(particles))
            permutate_tree(tree, particles)
            bs = time.time() - t
            bsl_temp.append(bs)

            t = time.time()
            tree = Node(np.zeros(3), 1e21, np.array(particles))
            with Pool() as pool:
                particles = pool.starmap(permutate_tree_v_multi, [(p, tree) for p in particles])
            for p in particles:
                permutate_pos_multi(p)
            bp = time.time() - t
            bpl_temp.append(bp)

        fsl.append(np.mean(fsl_temp))
        fpl.append(np.mean(fpl_temp))
        bsl.append(np.mean(bsl_temp))
        bpl.append(np.mean(bpl_temp))
        print('Brute Force serial: ' + str(fsl[-1]))
        print('Brute Force parallel: ' + str(fpl[-1]))
        print('Barnes Hut Serial: ' + str(bsl[-1]))
        print('Barnes Hut Parallel: ' + str(bpl[-1]))

    with open('comparison_data', 'w') as f:
        json.dump([fsl, fpl, bsl, bpl], f)

    mpl.rcParams["font.size"] = 15
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set(xlabel='Number of Partices', ylabel='Time required to permute the system one timestep [s]')
    plt.gcf().set_tight_layout(True) # To prevent the xlabel being cut off
    ax.set_xscale('log')
    ax.plot(nums, fsl, label='Brute Force Serial')
    ax.plot(nums, fpl, label='Brute Force Parallel')
    ax.plot(nums, bsl, label='Barnes Hut Serial')
    ax.plot(nums, bpl, label='Barnes Hut Parallel')
    plt.legend()
    plt.show()

                    