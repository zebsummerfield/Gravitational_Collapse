"""
Implementation of the barnes-hut approximation.
"""

import numpy as np

class Node:

    def __init__(self, centre: list, length: float, particles: np.array, oct=False) -> None:
        self.centre = centre
        self.length = length
        self.children = []

        if len(particles) == 1:
            self.mass = particles[0].mass
            self.com = particles[0].pos

        else:
            if oct:
                self.gen_child_oct(particles)
            else:
                self.gen_child_quad(particles)
            self.mass = sum([c.mass for c in self.children])
            self.com = sum([c.mass * c.pos for c in self.children]) / self.mass

    
    def gen_child_quad(self, particles: np.array):
        for i in range(2):
            for j in range(2):
                    child_centre = self.centre + 0.5*self.length*(np.array([i,j,0.5])-0.5)
                    child_particles = particles[[x >= self.centre/2 for x in particles.pos][:2] ==[i,j]]
                    if len(child_particles) > 0:
                        self.children.append(child_centre, self.length/2, child_particles)
    
    def gen_child_oct(self, particles: np.array):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    child_centre = self.centre + 0.5*self.length*(np.array([i,j,k])-0.5)
                    child_particles = particles[[x >= self.centre/2 for x in particles.pos] ==[i,j,k]]
                    if len(child_particles) > 0:
                        self.children.append(child_centre, self.length/2, child_particles, oct=True)
    


