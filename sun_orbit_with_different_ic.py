"""
plots the orbits of the [sun, earth, jupiter] system in seperate plots.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from particle import Particle
import initial_conditions as ic
import pdb

dt = 10 * ic.dt
sun1_particles = ic.create_particles(dt=dt)
sun1_particles[0] = Particle(1.9889e30, [0,0,0], [0,0,0], dt=dt, name='orbit with stationary intial conditions')
sun2_particles = ic.create_particles(dt=dt)
sun2_particles[0].name = 'orbit with updated initial conditions'
year_in_s = ic.year_in_s
sun_radius = 6.9570e8

mpl.rcParams["font.size"] = 25
fig, ax = plt.subplots(figsize=(6,10))
ax.set(ylim=[-60,20], xlim=[-20,20], xlabel='x [$R_{\odot}$]', ylabel='y [$R_{\odot}$]')
plt.gcf().set_tight_layout(True) # To prevent the xlabel being cut off

for particles in [sun1_particles, sun2_particles]:
    x_pos = [particles[0].pos[0]]
    y_pos = [particles[0].pos[1]]

    for i in range(int(year_in_s/dt)*100):
        for p in particles:
            p.calc_next_v(particles)
        for p in particles:
            p.set_new_v()
            p.calc_next_pos()
            p.set_new_pos()
        x_pos.append(particles[0].pos[0])
        y_pos.append(particles[0].pos[1])
    
    ax.plot(np.array(x_pos) / sun_radius, np.array(y_pos) / sun_radius, label = particles[0].name)

plt.show()
