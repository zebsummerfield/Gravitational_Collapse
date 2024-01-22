"""
Generating a galaxy of stars distributed normally.
"""

import numpy as np
import matplotlib.pyplot as plt
import distributions
import matplotlib.pyplot as plt

solar_mass = 2e30
galaxy_mass = 1e12 * solar_mass
num = 1000
mega_particle_mass = galaxy_mass / num
galaxy_radius = 5e20
dt = 1000000 * 365 * 24 * 3600
pc = 3.086e16
scale_length = 3 * 1000 * pc

# positions = distributions.normal_circle(galaxy_mass, galaxy_radius, num)
positions = distributions.disc_exp(galaxy_mass, scale_length, num)

fig, ax = plt.subplots(figsize=(10,10))
plot = ax.plot(positions[:,0], positions[:,1], marker='o', linestyle='none', markersize=2)

plt.show()



