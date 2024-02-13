"""
File contains the functions which create the intial conditions for several distributions.
"""
import random
import numpy as np
from particle import Particle
from velocities_analytic import *
from utils import modulus


G = 6.67*10**-11

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
            distance = - Rh * np.log(1 - random.uniform(0,1))
            if distance > 3 * Rh or distance == 0:
                continue
            x, y = (np.cos(delta) * distance, np.sin(delta) * distance)
            if all(np.sqrt((x-p[0])**2 + (y-p[1])**2) > 1e16 for p in positions):
                ok_dist = True

        velocity = gen_v_wcm(distance, density0, Rh, mass_c)
        vel = [-np.sin(delta) * velocity, np.cos(delta) * velocity, 0]
        pos = [x, y, 0]
        positions.append(pos)
        particles.append(Particle(mega_particle_mass, pos, vel, dt=dt))

    particles = np.array(particles)
    for p in particles:
        r = modulus(p.pos)
        if r > 0:
            p.set_circ_v(particles, density0, Rh)
            p.v += (p.v / modulus(p.v)) * np.random.normal(0, gen_v_dispersion_azimuthal(r, density0, Rh, mass_c, h=h))
            p.v += (p.pos / modulus(p.pos)) * np.random.normal(0, gen_v_dispersion_radial(r, density0, Rh, mass_c, h=h))

    return particles




