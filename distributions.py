"""
File contains the functions which create the intial conditions for several distributions.
"""
import random
import numpy as np
from particle import Particle
import velocities_analytic
from utils import modulus
import scipy.special as sp
import analytics_total

G = 6.670*10**-11

def normal_circle(mass: float, r: float, num: int) -> np.array:
    """
    Returns a list of num normally distributed positions within a circle of radius r.
    """
    positions = np.zeros((num,4))
    for i in range(num):
        delta = 2 * np.pi * random.random()
        distance = r * (random.random() ** 0.5)
        x, y = (np.cos(delta) * distance, np.sin(delta) * distance)
        velocity = np.sqrt(G * mass * distance / r**2)
        positions[i] = [x, y, -np.sin(delta) * velocity, np.cos(delta) * velocity]
    return positions 

def disc_exp(mass:float, Rh:float, num: int, dt: float, mass_c=0, h=1.2) -> np.array:
    """
    Returns a list of num particles distributed according to n = exp(-r/R) where r is the distance from the centre and R is the scale length.
    If mass_c is given then a central particle is also created with that mass.
    A velocity dispersion is applied to the disk where h is the Toomre stability parameter.
    """
    
    mega_particle_mass = (mass / num) * 1
    density0 = mass / (2 * np.pi * Rh**2)
    
    positions = []
    particles = []
    if mass_c:
        particles.append(Particle(mass_c, [0,0,0], [0,0,0], dt=dt))
        
    for i in range(num):

        ok_dist = False
        while not ok_dist:
            delta = 2 * np.pi * random.random()
            distance = - Rh * np.log(1 - random.random())
            if distance > 3 * Rh or distance == 0:
                continue
            x, y = (np.cos(delta) * distance, np.sin(delta) * distance)
            if all(np.sqrt((x-p[0])**2 + (y-p[1])**2) > 1e16 for p in positions):
                ok_dist = True

        velocity = velocities_analytic.gen_v_wcm(distance, density0, Rh, mass_c)
        vel = [-np.sin(delta) * velocity, np.cos(delta) * velocity, 0]
        pos = [x, y, 0]
        positions.append(pos)
        particles.append(Particle(mega_particle_mass, pos, vel, dt=dt))

    particles = np.array(particles)
    for p in particles:
        r = modulus(p.pos)
        if r > 0:
            p.set_circ_v(particles, density0, Rh)
            p.v += (p.v / modulus(p.v)) * np.random.normal(0, velocities_analytic.gen_v_dispersion_azimuthal(r, density0, Rh, mass_c, h=h))
            p.v += (p.pos / modulus(p.pos)) * np.random.normal(0, velocities_analytic.gen_v_dispersion_radial(r, density0, Rh, mass_c, h=h))

    return particles

def halo_NFW_mass(radius: float, Rh: float) -> float:
    qc = radius/Rh
    return np.log(1 + qc) - qc / (1 + qc)

def halo_NFW_positions(Rh: float, Rvir: float, num: int) -> np.array:
    M1 = halo_NFW_mass(Rvir, Rh)
    positions = np.zeros((num,3))
    for i in range(num):
        phi = 2 * np.pi * random.random()
        theta = np.arccos(1 - 2 * random.random())
        r = -Rh * (1 + 1 / (sp.lambertw(-np.exp(- M1 * random.random() - 1))))
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        positions[i] = (x.real ,y.real, z.real)
    return positions

def halo_NFW(mass:float, Rh: float, Rvir: float, num: int, dt: float) -> np.array:
    mega_particle_mass = (mass / num) * 1
    positions = halo_NFW_positions(Rh, Rvir, num)
    particles = []
    for i in range(num):
        particles.append(Particle(mega_particle_mass, positions[i], [0,0,0], dt=dt))
    particles = np.array(particles)

    for p in particles:
        v = np.sqrt(0.5 * abs(p.calc_potential_energy(particles)) / p.mass)
        v_dir = np.random.normal(0, 1, 3)
        v_hat = v_dir / np.sqrt(sum(v_dir * v_dir))
        p.v = v * v_hat
    return particles

def disc_all(mass: float, Rh: float, Rh_halo: float, c_mass: float, density0: float, rho0: float, num: int, dt: float) -> np.array:
    mega_particle_mass = (mass / num) * 1
    density0 = mass / (2 * np.pi * Rh**2)

    particles = []
    for i in range(num):
        delta = 2 * np.pi * random.random()
        distance = - Rh * np.log(1 - random.random())
        while distance > 5 * Rh or distance == 0:
            distance = - Rh * np.log(1 - random.random())
        pos = [np.cos(delta) * distance, np.sin(delta) * distance, 0]

        velocity = analytics_total.v_disc(distance, Rh, Rh_halo, c_mass, density0, rho0)
        vel = [-np.sin(delta) * velocity, np.cos(delta) * velocity, 0]

        particles.append(Particle(mega_particle_mass, pos, vel, dt=dt))

    return np.array(particles)
