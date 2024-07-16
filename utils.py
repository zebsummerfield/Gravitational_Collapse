"""
Functions needed in the project.
"""

import numpy as np
from barneshut import Node

def modulus(vector: np.array) -> float:
		"""
		Returns the absolute value of a vector.

		Arguments:
		---
		vector (numpy.array): The vector quantity.

		Returns:
		---
		total (float): The absolute value.
		"""
		total = np.sqrt(sum(vector[i]**2 for i in range(len(vector)))) 
		return total

def permutate(particles: np.array):
	"""
	Permutes the positions and velocities of a system of particles using the serial brute force method.
	"""
	for p in particles:
		p.calc_next_v(particles)
		p.set_new_v()
	for p in particles:
		p.calc_next_pos()
		p.set_new_pos()

def permutate_v_multi(target, particles: np.array):
	"""
	Permutes the velocities of a system of particles using the parallel brute force method.
	"""
	target.calc_next_v(particles)
	target.set_new_v()
	return target

def permutate_tree(tree: Node, particles: np.array):
	"""
	Permutes the positions and velocities of a system of particles using the serial Barnes-Hut method.
	"""
	for p in particles:
		p.calc_next_v_tree(tree)
		p.set_new_v()
	for p in particles:
		p.calc_next_pos()
		p.set_new_pos()

def permutate_tree_v_multi(target, tree: Node):
	"""
	Permutes the velocities of a system of particles using the parallel Barnes-Hut method.
	"""
	target.calc_next_v_tree(tree)
	target.set_new_v()
	return target

def permutate_pos_multi(target):
	"""
	Permutes the positions of a system of particles using the parallel method.
	"""
	target.calc_next_pos()
	target.set_new_pos()
	return target

def COM(particles):
	"""
	Returns the centre of mass of a system of particles.
	"""
	COM = np.zeros(3)
	M = 0
	for p in particles:
		COM += p.mass * p.pos
		M += p.mass
	return COM / M
