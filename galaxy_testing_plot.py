"""
Generating a galaxy of stars distributed normally.
"""

import numpy as np
import matplotlib.pyplot as plt
import distributions
import matplotlib as mpl

solar_mass = 2e30
galaxy_mass = 1e11 * solar_mass
num = 1000
mega_particle_mass = (galaxy_mass / num) * 1
year_in_s = 365 * 24 * 3600
dt = 100000 * year_in_s
pc = 3.086e16
scale_length = 3 * 1000 * pc
density0 = galaxy_mass / (2 * np.pi * scale_length**2)

# positions = distributions.normal_circle(galaxy_mass, galaxy_radius, num)
particles = np.array(distributions.disc_exp(galaxy_mass, scale_length, num, dt=dt, mass_c=0.2*galaxy_mass, h=1))
positions = np.zeros((len(particles), 3))
for index, p in enumerate(particles):
    positions[index] = p.pos

mpl.rcParams["font.size"] = 15
fig, ax = plt.subplots(figsize=(10,10))
ax.plot(positions[:,0]/(1000*pc), positions[:,1]/(1000*pc), marker='o', linestyle='none', markersize=1.5)
ax.set_xlabel('[$kpc$]')
ax.set_ylabel('[$kpc$]')
ax.set_xlim(-9,9)
ax.set_ylim(-9,9)
plt.show()



