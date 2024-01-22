"""
Animates the orbits of the [sun, earth, jupiter] system in seperate plots.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import initial_conditions as ic

year_in_s = ic.year_in_s
dt = 10 * ic.dt
particles = ic.create_particles(dt=dt)

fig, ax = plt.subplots(1, 4, figsize=(20,5))

x_pos = {}
y_pos = {}
for p in particles:
	x_pos[p] = [p.pos[0]] 
	y_pos[p] = [p.pos[1]]

sun_orbit = ax[0].plot(x_pos[particles[0]], y_pos[particles[0]], '-', linewidth=1, color='orange',
	markevery=[-1], marker='.', markersize=30)[0]
earth_orbit = ax[1].plot(x_pos[particles[1]], y_pos[particles[1]], '-', linewidth=1, color='blue',
	markersize=10, markevery=[-1], marker='.')[0]
jupiter_orbit = ax[2].plot(x_pos[particles[2]], y_pos[particles[2]], '-', linewidth=1, color='red',
	markersize=20, markevery=[-1], marker='.')[0]
orbits = [sun_orbit, earth_orbit, jupiter_orbit]

ax[0].set(xlim=[-1e9, 1e9], ylim=[-1e9, 1e9], title='Sun Orbit')
ax[1].set(xlim=[-2e11, 2e11], ylim=[-2e11, 2e11], title='Earth Orbit')
ax[2].set(xlim=[-1e12, 1e12], ylim=[-1e12, 1e12], title='Jupiter Orbit')
ax[3].set_axis_off()
time_text = ax[3].text(0.1, 0.5, 'Time Passed = 0 days', fontsize=15)

def update(frame):
	for p in particles:
		p.calc_next_v(particles)
	for index, p in enumerate(particles):
		p.set_new_v()
		p.calc_next_pos()
		p.set_new_pos()
		x_pos[p].append(p.pos[0])
		y_pos[p].append(p.pos[1])
		orbits[index].set_xdata(x_pos[p])
		orbits[index].set_ydata(y_pos[p])
	time_text.set_text(f'Time Passed = {round((frame+1) * dt/year_in_s, 3)} years')
	return (sun_orbit, earth_orbit, jupiter_orbit, time_text)

ani = animation.FuncAnimation(fig, update,  frames=10000, interval=5, blit=True)
plt.show()
