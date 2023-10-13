"""
plots the orbits of the [sun, earth, jupiter] system in seperate plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from particle import Particle
import initial_conditions as ic

dt = 10 * ic.dt
particles = ic.create_particles(dt=dt)
year_in_s = ic.year_in_s

fig, ax = plt.subplots(1, len(particles),figsize=(15,5))

x_pos = {}
y_pos = {}
for p in particles:
	x_pos[p] = [p.pos[0]] 
	y_pos[p] = [p.pos[1]]

print(int(year_in_s/dt)*100)

for i in range(int(year_in_s/dt)*100):
	for p in particles:
		p.calc_next_v(particles)
	for p in particles:
		p.set_new_v()
		p.calc_next_pos()
		p.set_new_pos()
		x_pos[p].append(p.pos[0])
		y_pos[p].append(p.pos[1])

for i, p in enumerate(particles):
	ax[i].plot(x_pos[p], y_pos[p])
	ax[i].set(title=p.name)
plt.show()