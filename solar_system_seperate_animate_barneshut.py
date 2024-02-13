"""
Animates the orbits of the [sun, earth, jupiter] system in seperate plots using the Barnes-Hut approximation.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from barneshut import Node
import numpy as np
from utils import *
from particle import Particle

year_in_s = 365.256*24*3600
dt = 10 * 24*3600
jupiter_period = year_in_s * 11.86
earth_v = 2 *np.pi * 1.4960e11 / year_in_s
jupiter_v = 2 * np.pi * 7.7854e11 / jupiter_period
sun_v_from_e = 5.9742e24 * earth_v / 1.9889e30
sun_v_from_j = 1.8986e27 * jupiter_v / 1.9889e30

def create_particles(dt=dt) -> list:
	"""
	Returns a list of particles [sun, earth, jupiter] created using the Particle class,
	with specific initial conditions.
	"""
	sun = Particle(1.9889e30, [1.8986e27*7.7854e11/1.9889e30 - 5.9742e24*1.4960e11/1.9889e30,0,0],
		[np.sin((dt/2)*(2*np.pi/jupiter_period))*sun_v_from_j - np.sin((dt/2)*(2*np.pi/year_in_s))*sun_v_from_e,
		np.cos((dt/2)*(2*np.pi/jupiter_period))*sun_v_from_j - np.cos((dt/2)*(2*np.pi/year_in_s))*sun_v_from_e, 0],
		dt=dt, name='sun')
	earth = Particle(5.9742e24, [1.4960e11,0,0],
		[np.sin((dt/2)*(2*np.pi/year_in_s))*earth_v,
		np.cos((dt/2)*(2*np.pi/year_in_s))*earth_v, 0], dt=dt, name='earth')
	jupiter = Particle(1.8986e27, [-7.7854e11,0,0],
		[-np.sin((dt/2)*(2*np.pi/jupiter_period))*jupiter_v,
		-np.cos((dt/2)*(2*np.pi/jupiter_period))*jupiter_v, 0], dt=dt, name='jupiter')
	return [sun, earth, jupiter]

particles = np.array(create_particles())
x_pos = {}
y_pos = {}
for p in particles:
	x_pos[p] = [p.pos[0]] 
	y_pos[p] = [p.pos[1]]

fig, axes = plt.subplots(1, 4, figsize=(20,5))
sun_orbit = axes[0].plot(x_pos[particles[0]], y_pos[particles[0]], '-', linewidth=1, color='orange',
	markevery=[-1], marker='.', markersize=30)[0]
earth_orbit = axes[1].plot(x_pos[particles[1]], y_pos[particles[1]], '-', linewidth=1, color='blue',
	markersize=10, markevery=[-1], marker='.')[0]
jupiter_orbit = axes[2].plot(x_pos[particles[2]], y_pos[particles[2]], '-', linewidth=1, color='red',
	markersize=20, markevery=[-1], marker='.')[0]
orbits = [sun_orbit, earth_orbit, jupiter_orbit]

axes[0].set(xlim=[-1e9, 1e9], ylim=[-1e9, 1e9], title='Sun Orbit')
axes[1].set(xlim=[-2e11, 2e11], ylim=[-2e11, 2e11], title='Earth Orbit')
axes[2].set(xlim=[-1e12, 1e12], ylim=[-1e12, 1e12], title='Jupiter Orbit')
axes[3].set_axis_off()
time_text = axes[3].text(0.1, 0.5, 'Time Passed = 0 days', fontsize=15)

def update(frame):
	tree = Node(np.zeros(3), 1e12, particles)
	permutate_tree(tree, particles)
	for index, p in enumerate(particles):
		x_pos[p].append(p.pos[0])
		y_pos[p].append(p.pos[1])
		orbits[index].set_xdata(x_pos[p])
		orbits[index].set_ydata(y_pos[p])
	time_text.set_text(f'Time Passed = {round((frame+1) * dt/year_in_s, 1):.1f} years')
	return (sun_orbit, earth_orbit, jupiter_orbit, time_text)

ani = animation.FuncAnimation(fig, update,  frames=10000, interval=10, blit=True)
plt.show()
