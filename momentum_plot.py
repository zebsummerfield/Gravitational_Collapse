"""
Plots the seperation and total momentum evolutions of the [sun, earth, jupiter] system.
"""

import numpy as np
import matplotlib.pyplot as plt
from particle import Particle
import initial_conditions as ic

year_in_s = ic.year_in_s
dt = 10 * ic.dt
particles = ic.create_particles(dt=dt)
time = 0
sun = particles[0]
earth = particles[1]

def modulus(vector):
		total = np.sqrt(sum(vector[i]**2 for i in range(3))) 
		return total

fig, ax = plt.subplots(1, 2, figsize=(10,5))
ax[0].set(ylim=[1.45e11,1.55e11], xlabel='Time [yrs]', ylabel='Seperation [m]')
ax[1].set(xlabel='Time [yrs]', ylabel='Momentum [kgms-1]')

seperation = [modulus(earth.pos - sun.pos)]
start_momentum = [0,0,0]
for p in particles:
	start_momentum += p.calc_momentum()
total_momentum = [modulus(start_momentum)]
time_tracker = [0]

for i in range(int(year_in_s/dt)*100):
	time += dt
	time_tracker.append(time)
	momentum = 0
	for p in particles:
		p.calc_next_v(particles)
	for p in particles:
		p.set_new_v()
		p.calc_next_pos()
		p.set_new_pos()
		momentum += p.calc_momentum()
	seperation.append(modulus(earth.pos - sun.pos))
	total_momentum.append((modulus(momentum)))

time_tracker = np.array(time_tracker) / year_in_s
	
ax[0].plot(time_tracker, seperation)
ax[1].plot(time_tracker, total_momentum)

print(f"Change in Seperation = {abs((seperation[-1]-seperation[0])/seperation[0])*100}%")

plt.show()

