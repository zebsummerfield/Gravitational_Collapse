"""
File containing parameter constants for the simulations.
"""
import numpy as np

G = 6.674*10**-11
pc = 3.086e16
epsilon = 1000 * pc
theta = 0.5
solar_mass = 2e30
disc_mass = 1e11 * solar_mass
centre_mass = 2e10 * solar_mass
halo_mass = 1e12 * solar_mass
year_in_s = 365 * 24 * 3600
scale_radius = 3 * 1000 * pc
halo_scale_radius = 20 * 1000 * pc
virial_radius = 200 * 1000 * pc
disc_density0 = disc_mass / (2 * np.pi * scale_radius**2)
halo_density0 = halo_mass / (4 * np.pi * halo_scale_radius**3 * (np.log(1 + virial_radius/halo_scale_radius) - virial_radius/(virial_radius+halo_scale_radius)))
