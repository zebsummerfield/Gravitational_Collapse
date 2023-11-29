
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
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

mpl.rcParams["font.size"] = 25
fig, ax = plt.subplots(1, 2, figsize=(20,10))
ax[0].set(ylim=[-1,1], xlabel='Time [ $yrs$ ]', ylabel='Percentage Change in Total Energy')
ax[1].set(ylim=[0,3e-15], xlabel='Time [ $yrs$ ]', ylabel='Total Momentum / Total Momentum Magnitude')
plt.gcf().set_tight_layout(True) # To prevent the xlabel being cut off

seperation = [modulus(earth.pos - sun.pos)]
KE = [0]
PE = [0]
total_energy = [0]
initial_linear_momentum = 0
start_momentum = [0,0,0]
for p in particles:
	KE[0] += p.calc_kinetic_energy()
	PE[0] += p.calc_potential_energy(particles)
	initial_linear_momentum += modulus(p.calc_momentum())
	start_momentum += p.calc_momentum()
total_energy[0] += np.float64(KE[0] + PE[0])
initial_linear_momentum = np.float64(initial_linear_momentum)
total_momentum = [modulus(start_momentum)]
time_tracker = [0]


for i in range(int(year_in_s/dt)*100):
	time += dt / year_in_s
	time_tracker.append(time)
	ke = 0
	pe = 0
	momentum = 0
	for p in particles:
		p.calc_next_v(particles)
	for p in particles:
		p.set_new_v()
		p.calc_next_pos()
		p.set_new_pos()
		ke += p.calc_kinetic_energy()
		pe += p.calc_potential_energy(particles)
		momentum += p.calc_momentum()
	seperation.append(modulus(earth.pos - sun.pos))
	KE.append(ke)
	PE.append(pe)
	total_energy.append(ke+pe)
	total_momentum.append((modulus(momentum)))
	
ax[0].plot(time_tracker, ((total_energy - total_energy[0]) / total_energy[0]) * 100, color='green', linewidth=5)
ax[1].plot(time_tracker, total_momentum / initial_linear_momentum, color='red', linewidth=1)

plt.show()