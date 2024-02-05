import numpy as np
from particle import Particle
import distributions
from utils import *
import time
from barneshut import Node
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib.patches as patches

solar_mass = 2e30
galaxy_mass = 1e12 * solar_mass
num = 1000
mega_particle_mass = (galaxy_mass / num) * 0.5
galaxy_radius = 5e20
dt = 10000 * 365 * 24 * 3600
pc = 3.086e16
scale_length = 3 * 1000 * pc

positions = distributions.normal_circle(galaxy_mass, galaxy_radius, num)
# positions = distributions.disc_exp(galaxy_mass, scale_length, num)
particles = []
for pos in positions:
    particles.append(Particle(mega_particle_mass, [pos[0], pos[1], 0], [pos[2], pos[3], 0], dt=dt))
particles = np.array(particles)

# t = time.time()
# permutate(particles)
# print(time.time() - t)

fig, ax = plt.subplots(figsize=(10,10))
ax.set_axis_off()
ax.plot(positions[:,0], positions[:,1], marker='o', linestyle='none', markersize=2)
ax.plot(particles[0].pos[0], particles[0].pos[1], marker='o', markersize=5, color="lime")

t = time.time()
tree = Node(np.zeros(3), 1e21, particles)
print(time.time() - t)


def add_rect_to_plot(node: Node, ax: Axes):
    square = patches.Rectangle((node.centre[0]-(node.length/2), node.centre[1]-(node.length/2)), node.length, node.length, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(square)
    for c in node.children:
        add_rect_to_plot(c, ax)

def add_children_rect_to_plot(node: Node, ax: Axes):
    if node.children:
        for c in node.children:
            add_children_rect_to_plot(c, ax)
    else:
        square = patches.Rectangle((node.centre[0]-(node.length/2), node.centre[1]-(node.length/2)), node.length, node.length, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(square)

def add_target_rect_to_plot(p: Particle, node: Node, ax: Axes):
    relative_pos = (p.pos - node.com)
    d = modulus(relative_pos)
    if d > 0:
        if len(node.children) == 0 or node.length / d < 0.5:
            square = patches.Rectangle((node.centre[0]-(node.length/2), node.centre[1]-(node.length/2)), node.length, node.length, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(square)
        else:
            for c in node.children:
                add_target_rect_to_plot(p, c, ax)

# add_rect_to_plot(tree, ax)
# add_children_rect_to_plot(tree, ax)
add_target_rect_to_plot(particles[0], tree, ax)

t = time.time()
permutate_tree(tree, particles)
print(time.time() - t)
                
plt.show()