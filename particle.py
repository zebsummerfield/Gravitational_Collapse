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
		self.epsilon = 2e19
		self.mass = mass
		self.pos = np.array(initial_pos)
		self.v = np.array(initial_v)
		self.dt = dt
		self.name = name
		self.field_no_G = np.zeros(3)
		self.potential_no_G = 0
		self.sigma_v = 0

	def force(self, particles: list, r=None) -> np.array:
		if type(r) == type(None):
			r = self.pos
		fieldnoG = np.zeros(3)
		for p in particles:
			if p is not self:
				relative_pos = (r - p.pos)
				d = modulus(relative_pos)
				fieldnoG += p.mass * relative_pos / (np.sqrt(d**2 + self.epsilon**2) ** 3)
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
	
	def tree_walk(self, node: Node):
		relative_pos = (self.pos - node.com)
		d = modulus(relative_pos)
		if d > 0:
			if len(node.children) == 0 or node.length / d < 0.5:
				self.field_no_G += node.mass * relative_pos / (np.sqrt(d**2 + self.epsilon**2) ** 3)
			else:
				for child in node.children:
					self.tree_walk(child)
	
	def calc_next_v_tree(self, tree: Node) -> np.array:
		force = self.force_tree(tree)
		self.next_v = self.v + (force * self.dt / self.mass)
		return self.next_v
	
	def set_new_v(self):
		self.v = self.next_v
	
	def set_circ_v(self, particles: list, density0: float, Rh: float, h=1.2, dr=1e16):
		r = modulus(self.pos)
		if r == 0:
			self.v = np.zeros(3)
		else:
			condition = np.array([modulus(p.pos) < modulus(self.pos) for p in particles])
			force = self.force(particles[condition])
			rad_force = abs(np.dot(force, self.pos) / r)
			v = np.sqrt(r * rad_force / self.mass)
			v_dir = np.cross(self.pos/r, [0,0,1]) 
			self.v = v * v_dir * 1
			# sigma_v = self.v_dispersion(particles[condition], density0, Rh, h, dr)
			# self.sigma_v = sigma_v
			# theta = 2*np.pi * np.random.random()
			# self.v += np.array([np.cos(theta), np.sin(theta), 0]) * np.random.normal(0, sigma_v)

	def v_dispersion(self, particles: list, density0: float, Rh: float, h: float, dr: float) -> float:
		r = modulus(self.pos)
		term1 = 3 * modulus(self.v)**2 / r**2 
		f1 = abs(np.dot(self.force(particles, self.pos + (dr * self.pos / r)), self.pos) / r)
		f2 = abs(np.dot(self.force(particles, self.pos + (dr - self.pos / r)), self.pos) / r)
		term2 = (f1 - f2) / (2*dr * self.mass)
		if term1 + term2 > 0:
			epicyclic_frequency = np.sqrt(term1 + term2)
		else:
			return 0
		density = density0 * np.exp(-r/Rh)
		return 3.36 * h * self.G * density / epicyclic_frequency
	
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
				PE += - self.G * self.mass * p.mass / np.sqrt(modulus(relative_pos)**2 + self.epsilon**2)
		return PE
	
	def potential_tree(self, tree: Node) -> float:
		self.potential_no_G = 0
		self.tree_walk_potential(tree)
		PE = self.potential_no_G * (- self.G * self.mass)
		return PE
	
	def tree_walk_potential(self, node: Node):
		relative_pos = (self.pos - node.com)
		d = modulus(relative_pos)
		if d > 0:
			if len(node.children) == 0 or node.length / d < 0.5:
				self.potential_no_G += node.mass / np.sqrt(d**2 + self.epsilon**2)
			else:
				for child in node.children:
					self.tree_walk(child)
	
	def calc_total_energy(self, tree: Node) -> float:
		return self.calc_kinetic_energy() + self.potential_tree(tree)

	def calc_momentum(self) -> np.array:
		momentum = self.mass * self.v
		return momentum



