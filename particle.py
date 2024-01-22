"""
File containing the definition of the Particle class.
"""

import numpy as np
from utils import modulus
from barneshut import Node

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
		self.epsilon = 1e16
		self.mass = mass
		self.pos = np.array(initial_pos)
		self.v = np.array(initial_v)
		self.dt = dt
		self.name = name
		self.field_no_G = np.zeros(3)

	def force(self, particles: list) -> np.array:
		fieldnoG = np.zeros(3)
		for p in particles:
			if p is not self:
				relative_pos = (self.pos - p.pos)
				d = modulus(relative_pos)
				fieldnoG += p.mass * relative_pos / ((d**2 + self.epsilon**2) ** (3/2))
		force = fieldnoG * (- self.G * self.mass)
		return force
		
	def calc_next_v(self, particles: list) -> np.array:
		force = self.force(particles)
		self.next_v = self.v + (force * self.dt / self.mass)
		return self.next_v
	
	def force_tree(self, tree: Node) -> np.array:
		self.field_no_G = np.zeros(3)
		self.tree_walk(tree)
		force = self.field_no_G * (- self.G * self.mass)
		return force
	
	def tree_walk(self, node: Node) -> np.array:
		relative_pos = (self.pos - node.com)
		d = modulus(relative_pos)
		if len(node.children) == 0 or node.length / d < 0.5:
			self.field_no_G += node.mass * relative_pos / ((d**2 + self.epsilon**2) ** (3/2))
		else:
			for c in node.children:
				self.tree_walk(c)
	
	def calc_next_v_tree(self, tree: Node) -> np.array:
		force = self.force_tree(tree)
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
		KE = 0.5 * self.mass * (modulus(self.v))**2
		return KE
	
	def calc_potential_energy(self, particles: list) -> float:
		half_pos = self.calc_half_pos()
		PE = 0
		for p in particles:
			if p is not self:
				relative_pos = (half_pos - p.pos)
				distance = modulus(relative_pos)
				PE += - self.G * self.mass * p.mass / distance
		return PE

	def calc_momentum(self) -> np.array:
		momentum = self.mass * self.v
		return momentum



