"""
Plots the orbits of the sun and the earth.
"""

import numpy as np
import matplotlib.pyplot as plt
from particle import Particle

time = 0
year_in_s = 365.256*24*3600
dt = 10*24*3600

earth_velocity = 2*np.pi*1.4960e11/year_in_s
sun_velocity = 5.9742e24 * earth_velocity / 1.9889e30
sun = Particle(1.9889e30, [0,0,0],
	[-np.sin((dt/2)*(2*np.pi/year_in_s))*sun_velocity,
  	-np.cos((dt/2)*(2*np.pi/year_in_s))*sun_velocity,0], dt=dt)
earth = Particle(5.9742e24, [1.4960e11,0,0],
	[np.sin((dt/2)*(2*np.pi/year_in_s))*earth_velocity,
  	np.cos((dt/2)*(2*np.pi/year_in_s))*earth_velocity,0], dt=dt)
particles = [sun, earth]

fig, ax = plt.subplots(1, len(particles),figsize=(10,5))

x_pos = {}
y_pos = {}
for p in particles:
	x_pos[p] = [p.pos[0]] 
	y_pos[p] = [p.pos[1]]

print(x_pos)
print(y_pos)

for i in range(int(year_in_s/dt)*100):
	time += dt
	for p in particles:
		p.calc_next_v(particles)
	for p in particles:
		p.set_new_v()
		p.calc_next_pos()
		p.set_new_pos()
		x_pos[p].append(p.pos[0])
		y_pos[p].append(p.pos[1])

for i, p in enumerate(particles):
	ax[i].scatter(x_pos[p], y_pos[p], marker='.', sizes=[2 for i in range(len(x_pos))])
plt.show()