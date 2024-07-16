"""
Analytical functions for a galactic disc.
"""

import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
from constants import *
import matplotlib as mpl

def potential_particle(r: float, mass: float, epsilon=epsilon) -> float:
    """Returns the potential at a distance r from a particle."""
    return - G * mass / np.sqrt(r**2 + epsilon**2)

def diff_potential_particle(r: float, mass: float, epsilon=epsilon) -> float:
    """Returns the differential of potential at a distance r from a particle."""
    return G * mass * r / (r**2 + epsilon**2)**(3/2)

def potential_disc(r: float, density0: float, Rh: float) -> float:
    """Returns the potential due to a disc at a distance r from the centre."""
    y = r / (2 * Rh)
    return - np.pi * G * density0 * r * (sp.i0(y) * sp.k1(y) - sp.i1(y) * sp.k0(y))

def diff_potential_disc(r: float, density0: float, Rh: float) -> float:
    """Returns the differential of potential due to a disc at a distance r from the centre."""
    y = r / (2 * Rh)
    return (4 * np.pi * G * density0 * Rh * y**2 * (sp.i0(y) * sp.k0(y) - sp.i1(y) * sp.k1(y))) / r

def potential_halo(r: float, rho0: float, Rh_halo: float) -> float:
    """Returns the potential due to a halo at a distance r from the centre."""
    return - 4 * np.pi * G * rho0 * Rh_halo**3 * np.log(1 + r/Rh_halo) / r

def diff_potential_halo(r: float, rho0: float, Rh_halo: float) -> float:
    """Returns the differential of potential due to a halo at a distance r from the centre."""
    return 4 * np.pi * G * rho0 * Rh_halo**3 * (np.log(1 + r/Rh_halo) - r/(r+Rh_halo)) / r**2

def total_potential(r: float, Rh: float, Rh_halo: float, c_mass: float, density0: float, rho0: float) -> float:
    """Returns the total potential due to a central particle, disc and halo at a distance r."""
    return potential_particle(r, c_mass) + potential_disc(r, density0, Rh) + potential_halo(r, rho0, Rh_halo)

def total_diff_potential(r: float, Rh: float, Rh_halo: float, c_mass: float, density0: float, rho0: float) -> float:
    """Returns the differential of total potential due to a central particle, disc and halo at a distance r."""
    return diff_potential_particle(r, c_mass) + diff_potential_disc(r, density0, Rh) + diff_potential_halo(r, rho0, Rh_halo)

def diff_diff_potential(r: float, Rh: float, Rh_halo: float, c_mass: float, density0: float, rho0: float, dx=0.5*pc) -> float:
    """Returns the second differential of total potential due to a central particle, disc and halo at a distance r."""
    return  (total_diff_potential(r+dx, Rh, Rh_halo, c_mass, density0, rho0) -  total_diff_potential(r-dx, Rh, Rh_halo, c_mass, density0, rho0)) / (2*dx)

def v_disc(r: float, Rh: float, Rh_halo: float, c_mass: float, density0: float, rho0: float) -> float:
    """Returns the velocity of a particle in a disc at a distance r from the centre."""
    return np.sqrt(r * total_diff_potential(r, Rh, Rh_halo, c_mass, density0, rho0))
    
def angular_v_disc(r: float, Rh: float, Rh_halo: float, c_mass: float, density0: float, rho0: float) -> float:
    """Returns the angular velocity of a particle in a disc at a distance r from the centre."""
    return v_disc(r, Rh, Rh_halo, c_mass, density0, rho0) / r

def epicyclic(r: float, Rh: float, Rh_halo: float, c_mass: float, density0: float, rho0: float) -> float:
    """Returns the epicyclic frequency of a particle's orbit at a distance r from the centre of the disc."""
    term1 = 3 * v_disc(r, Rh, Rh_halo, c_mass, density0, rho0)**2 / r**2
    term2 = diff_diff_potential(r, Rh, Rh_halo, c_mass, density0, rho0)
    return np.sqrt(term1 + term2)

def density_disc(r: float, density0: float, Rh: float) -> float:
    """Returns the mass density of the disc at a distance r from the centre."""
    return density0 * np.exp(-r/Rh)

def v_dispersion_radial(r: float, Rh: float, Rh_halo: float, c_mass: float, density0: float, rho0: float, Q=1.2) -> float:
    """Returns the radial velocity dispersion of the disc at a distance r from the centre."""
    return Q * 3.36 * G * density_disc(r, density0, Rh) / epicyclic(r, Rh, Rh_halo, c_mass, density0, rho0)

def v_dispersion_azimuthal(r: float, Rh: float, Rh_halo: float, c_mass: float, density0: float, rho0: float, Q=1.2) -> float:
    """Returns the azimuthal velocity dispersion of the disc at a distance r from the centre."""
    return v_dispersion_radial(r, Rh, Rh_halo, c_mass, density0, rho0, Q=Q) * epicyclic(r, Rh, Rh_halo, c_mass, density0, rho0) / (2 * angular_v_disc(r, Rh, Rh_halo, c_mass, density0, rho0))

def relaxation_time(r: float, Rh: float, Rh_halo: float, c_mass: float, density0: float, rho0: float, Q=1.2) -> float:
    return np.sqrt(v_dispersion_radial(r, Rh, Rh_halo, c_mass, density0, rho0, Q=Q) * v_dispersion_azimuthal(r, Rh, Rh_halo, c_mass, density0, rho0, Q=Q)) / (np.pi * G * density_disc(r, density0, Rh))

def relaxation_time_softened(r: float, Rh: float, Rh_halo: float, c_mass: float, density0: float, rho0: float, d_mass: float, num: int, Q=1.2, epsilon=epsilon) -> float:
    return epsilon * num * v_dispersion_radial(r, Rh, Rh_halo, c_mass, density0, rho0, Q=Q) * v_dispersion_azimuthal(r, Rh, Rh_halo, c_mass, density0, rho0, Q=Q) * relaxation_time(r, Rh, Rh_halo, c_mass, density0, rho0, Q=Q) / (G * d_mass)

if __name__ =='__main__':

    print((relaxation_time_softened(scale_radius, scale_radius, halo_scale_radius, centre_mass, disc_density0, halo_density0, disc_mass, 1501, Q=1, epsilon=epsilon) / (year_in_s*1000000)))

    distances = np.linspace(pc, 5*scale_radius, num=1000)
    disc_p = []
    halo_p = []
    particle_p = []
    total_p = []
    disc_v = []
    disc_dispersion_radial = []
    disc_dispersion_azimuthal = []
    disc_relaxation_time = []
    disc_relaxation_time_softened = []
    for distance in distances:
        #particle_p.append(diff_potential_particle(distance, centre_mass))
        #disc_p.append(diff_potential_disc(distance, disc_density0, scale_radius))
        #halo_p.append(diff_potential_halo(distance, halo_density0, halo_scale_radius))
        #total_p.append(total_diff_potential(distance, scale_radius, halo_scale_radius, centre_mass, disc_density0, halo_density0))
        disc_v.append(v_disc(distance, scale_radius, halo_scale_radius, centre_mass, disc_density0, halo_density0)/1000)
        disc_dispersion_radial.append(v_dispersion_radial(distance, scale_radius, halo_scale_radius, centre_mass, disc_density0, halo_density0)/1000)
        disc_dispersion_azimuthal.append(v_dispersion_azimuthal(distance, scale_radius, halo_scale_radius, centre_mass, disc_density0, halo_density0)/1000)
        #disc_relaxation_time.append(relaxation_time(distance, scale_radius, halo_scale_radius, centre_mass, disc_density0, halo_density0, Q=1) / (year_in_s*1000000))
        #disc_relaxation_time_softened.append(relaxation_time_softened(distance, scale_radius, halo_scale_radius, centre_mass, disc_density0, halo_density0, disc_mass, 1501, Q=1, epsilon=epsilon) / (year_in_s*1000000))
    mpl.rcParams["font.size"] = 18
    fig, ax = plt.subplots(figsize=(10,8))
    ax.set(xlabel='Distance From Galactic Centre [$kpc$]', ylabel='Velocity [$kms^{-1}$]')
    #ax.plot(distances/(1000*pc), particle_p, label='particle')
    #ax.plot(distances/(1000*pc), disc_p, label='disc')
    #ax.plot(distances/(1000*pc), halo_p, label='halo')
    #ax.plot(distances/(1000*pc), total_p, label='total')
    ax.plot(distances/(1000*pc), disc_v, label='Circular Velocity')
    ax.plot(distances/(1000*pc), disc_dispersion_radial, label='Radial Dispersion')
    ax.plot(distances/(1000*pc), disc_dispersion_azimuthal, label='Azimuthal Dispersion')
    #ax.plot(distances/(1000*pc), disc_relaxation_time, label='relaxation time')
    #ax.plot(distances/(1000*pc), disc_relaxation_time_softened, label='softened relaxation time')
    print((scale_radius, halo_scale_radius))
    plt.legend()
    plt.show()
