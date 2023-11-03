"""
plots the orbits of the [sun, earth, jupiter] system in seperate plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from particle import Particle
import initial_conditions as ic

dt = 10 * ic.dt
sun1_particles = ic.create_particles(dt=dt)
sun1_particles[0] = Particle(1.9889e30, [0,0,0], [0,0,0], dt=dt, name='orbit with stationary intial conditions')
sun2_particles = ic.create_particles(dt=dt)
sun2_particles[0].name = 'orbit with updated initial conditions'
year_in_s = ic.year_in_s

fig, ax = plt.subplots(figsize=(4,8), layout='tight')
ax.set(ylim=[-4e10,1e10], xlim=[-1e10,1e10], xlabel='x [ $m$ ]', ylabel='y [ $m$ ]')

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

    ax.plot(x_pos, y_pos, label = particles[0].name)

plt.show()
