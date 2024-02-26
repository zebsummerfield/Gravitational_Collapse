"""
Generates a galaxy of stars distributed normally.
"""
import numpy as np
import matplotlib.pyplot as plt
import distributions
import matplotlib as mpl
from constants import *

num = 1000
mega_particle_mass = (disc_mass / num) * 1
dt = 100000 * year_in_s

# positions = distributions.normal_circle(galaxy_mass, galaxy_radius, num)
particles = np.array(distributions.disc_exp(disc_mass, scale_radius, num, dt=dt, mass_c=centre_mass, h=1))
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



