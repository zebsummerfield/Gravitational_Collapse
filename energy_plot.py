"""
Plots the energy and seperation evolutions of the [sun, earth, jupiter] system.
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

fig, ax = plt.subplots(2, 2, figsize=(10,10))
ax[0][0].set(ylim=[1.45e11,1.55e11], xlabel='Time [yrs]', ylabel='Seperation [m]')
ax[1][0].set(ylim=[1.4e35,1.8e35], xlabel='Time [yrs]', ylabel='Kinetic Energy [J]')
ax[1][1].set(ylim=[-0.7e36,-0.4e36], xlabel='Time [yrs]', ylabel='Potential Energy [J]')
ax[0][1].set(ylim=[-0.7e36,-0.4e36], xlabel='Time [yrs]', ylabel='Total Energy [J]')

seperation = [modulus(earth.pos - sun.pos)]
KE = [0]
PE = [0]
total_energy = [0]
for p in particles:
	KE[0] += p.calc_kinetic_energy()
	PE[0] += p.calc_potential_energy(particles)
total_energy[0] += KE[0] + PE[0]
time_tracker = [0]

for i in range(int(year_in_s/dt)*100):
	time += dt
	time_tracker.append(time)
	ke = 0
	pe = 0
	for p in particles:
		p.calc_next_v(particles)
	for p in particles:
		p.set_new_v()
		p.calc_next_pos()
		p.set_new_pos()
		ke += p.calc_kinetic_energy()
		pe += p.calc_potential_energy(particles)
	seperation.append(modulus(earth.pos - sun.pos))
	KE.append(ke)
	PE.append(pe)
	total_energy.append(ke+pe)

time_tracker = np.array(time_tracker) / year_in_s
	
ax[0][0].plot(time_tracker, seperation)
ax[1][0].plot(time_tracker, KE)
ax[1][1].plot(time_tracker, PE)
ax[0][1].plot(time_tracker, total_energy)

print(f"Change in Energy = {abs((total_energy[-1]-total_energy[0])/total_energy[0])*100}%")
print(f"Change in Seperation = {abs((seperation[-1]-seperation[0])/seperation[0])*100}%")

plt.show()

