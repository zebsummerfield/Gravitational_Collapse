"""
Plots the energy and seperation evolutions of the [sun, earth, jupiter] system.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import initial_conditions as ic

year_in_s = ic.year_in_s
dt = np.array([50, 25, 10, 1]) * ic.dt
particle_sets = [ic.create_particles(dt=i) for i in dt]

def modulus(vector):
		total = np.sqrt(sum(vector[i]**2 for i in range(3))) 
		return total

mpl.rcParams["font.size"] = 25
fig, ax = plt.subplots(figsize=(10,10))
ax.set(xlabel='Time [yrs]', ylabel='Percentage change in Seperation')
plt.gcf().set_tight_layout(True) # To prevent the xlabel being cut off



for index, particles in enumerate(particle_sets):
    time = 0
    sun = particles[0]
    earth = particles[1]
    seperation = [modulus(earth.pos - sun.pos)]
    time_tracker = [0]

    for i in range(int(year_in_s / dt[index]) * 10):
        time += dt[index] / year_in_s
        time_tracker.append(time)
        for p in particles:
            p.calc_next_v(particles)
        for p in particles:
            p.set_new_v()
            p.calc_next_pos()
            p.set_new_pos()
        seperation.append(modulus(earth.pos - sun.pos))
        
    ax.plot(time_tracker, ((seperation - seperation[0]) / seperation[0]) * 100, label = f"{int(dt[index] / ic.dt)} day timestep", linewidth=4)

ax.legend()
plt.show()
