"""
File containing the definition of the Particle class.
"""

import numpy as np
import matplotlib.pyplot as plt
import math

class Particle():
	"""
	A class representing objects which behave as particles
	and can interact with each other through gravitational forces.
	"""

	def __init__(self, mass: float, initial_pos: list, initial_v: list, dt=24*3600, name=''):
		"""
		Initialises a Particle object.

		Parameters:
		---
		mass (float): The mass of the particle.
		initial_pos (list): The initial position of the particle.
		initial_v (list): The initial velocity of the particle.
		dt (float): The time step in the leap-frog simulation in seconds.
		name (str): The name of the particle.
		"""
		self.G = 6.67*10**-11
		self.mass = mass
		self.pos = np.array(initial_pos)
		self.v = np.array(initial_v)
		self.dt = dt
		self.name = name
	
	def modulus(self, vector: np.array):
		"""
		Returns the absolute value of a vector.

		Arguments:
		---
		vector (numpy.array): The vector quantity.

		Returns:
		---
		total (float): The absolute value.
		"""
		total = math.sqrt(sum(vector[i]**2 for i in range(3))) 
		return total

	def force(self, particles: list) -> np.array:
		force = np.zeros(3)
		for p in particles:
			if p is not self:
				relative_pos = (self.pos - p.pos)
				distance = self.modulus(relative_pos)
				force += - self.G * self.mass * p.mass * relative_pos / distance**3
		return force
	
	def calc_next_v(self, particles: list) -> np.array:
		force = self.force(particles)
		self.next_v = self.v + (force * self.dt / self.mass)
		return self.next_v

	def set_new_v(self):
		self.v = self.next_v
		
	def calc_next_pos(self) -> np.array:
		self.next_pos = self.pos + self.v * self.dt
		return self.next_pos
	
	def set_new_pos(self):
		self.pos = self.next_pos
		   
	def calc_half_pos(self) -> np.array:
		half_pos = self.pos + self.v * self.dt * 0.5
		return half_pos
		
	def calc_kinetic_energy(self) -> float:
		KE = 0.5 * self.mass * (self.modulus(self.v))**2
		return KE
	
	def calc_potential_energy(self, particles: list) -> float:
		half_pos = self.calc_half_pos()
		PE = 0
		for p in particles:
			if p is not self:
				relative_pos = (half_pos - p.pos)
				distance = self.modulus(relative_pos)
				PE += - self.G * self.mass * p.mass / distance
		return PE

	def calc_momentum(self) -> np.array:
		momentum = self.mass * self.v
		return momentum



