"""
Animates the orbits of the [sun, earth, jupiter] system in 1 plot.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import initial_conditions as ic

year_in_s = ic.year_in_s
dt = 10 * ic.dt
particles = ic.create_particles(dt=dt)
AU = 1.4960e11

mpl.rcParams["font.size"] = 20
fig, ax = plt.subplots(1, 2, figsize=(20,10))

x_pos = {}
y_pos = {}
for p in particles:
	x_pos[p] = [p.pos[0]/AU] 
	y_pos[p] = [p.pos[1]/AU]

sun_orbit = ax[0].plot(x_pos[particles[0]], y_pos[particles[0]], '-', linewidth=1, color='orange',
	markevery=[-1], marker='.', markersize=30)[0]
earth_orbit = ax[0].plot(x_pos[particles[1]], y_pos[particles[1]], '-', linewidth=1, color='blue',
	markersize=10, markevery=[-1], marker='.')[0]
jupiter_orbit = ax[0].plot(x_pos[particles[2]], y_pos[particles[2]], '-', linewidth=1, color='red',
	markersize=20, markevery=[-1], marker='.')[0]
orbits = [sun_orbit, earth_orbit, jupiter_orbit]

ax[0].set(xlim=[-8, 8], ylim=[-8, 8], title='Solar System Orbits', xlabel='x [AU]', ylabel='y [AU]')
ax[1].set_axis_off()
time_text = ax[1].text(0.1, 0.5, 'Time Passed = 0 days', fontsize=30)

def update(frame):
	for p in particles:
		p.calc_next_v(particles)
	for index, p in enumerate(particles):
		p.set_new_v()
		p.calc_next_pos()
		p.set_new_pos()
		x_pos[p].append(p.pos[0]/AU)
		y_pos[p].append(p.pos[1]/AU)
		orbits[index].set_xdata(x_pos[p])
		orbits[index].set_ydata(y_pos[p])
	time_text.set_text(f'Time Passed = {round((frame+1) * dt/year_in_s, 1):.1f} years')
	return (sun_orbit, earth_orbit, jupiter_orbit, time_text)

ani = animation.FuncAnimation(fig, update,  frames=10000, interval=5, blit=True)
plt.show()
