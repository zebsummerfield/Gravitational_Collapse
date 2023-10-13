"""
File contains general constants for use in orbit programs
and the function to create a the list of particles: [sun, earth , jupiter].
"""

import numpy as np
from particle import Particle

year_in_s = 365.256*24*3600
dt = 1*24*3600
jupiter_period = year_in_s * 11.86

earth_v = 2 *np.pi * 1.4960e11 / year_in_s
jupiter_v = 2 * np.pi * 7.7854e11 / jupiter_period
sun_v_from_e = 5.9742e24 * earth_v / 1.9889e30
sun_v_from_j = 1.8986e27 * jupiter_v / 1.9889e30

def create_particles(dt=dt) -> list:
	"""
	Returns a list of particles [sun, earth, jupiter] created using the Particle class,
	with specific initial conditions.
	"""

	#sun = Particle(1.9889e30, [0,0,0], [0,0,0], dt=dt, name='sun')
	#sun = Particle(1.9889e30, [-5.9742e24*1.4960e11/1.9889e30,0,0],
	#	[-np.sin((dt/2)*(2*np.pi/year_in_s))*sun_v_from_e,
  	#	-np.cos((dt/2)*(2*np.pi/year_in_s))*sun_v_from_e,0], dt=dt)
	#sun = Particle(1.9889e30, [1.8986e27*7.7854e11/1.9889e30,0,0],
	#	[np.sin((dt/2)*(2*np.pi/jupiter_period))*sun_v_from_j,
	#	np.cos((dt/2)*(2*np.pi/jupiter_period))*sun_v_from_j, 0], dt=dt, name='sun')
	sun = Particle(1.9889e30, [1.8986e27*7.7854e11/1.9889e30 - 5.9742e24*1.4960e11/1.9889e30,0,0],
		[np.sin((dt/2)*(2*np.pi/jupiter_period))*sun_v_from_j - np.sin((dt/2)*(2*np.pi/year_in_s))*sun_v_from_e,
		np.cos((dt/2)*(2*np.pi/jupiter_period))*sun_v_from_j - np.cos((dt/2)*(2*np.pi/year_in_s))*sun_v_from_e, 0],
		dt=dt, name='sun')
	earth = Particle(5.9742e24, [1.4960e11,0,0],
		[np.sin((dt/2)*(2*np.pi/year_in_s))*earth_v,
		np.cos((dt/2)*(2*np.pi/year_in_s))*earth_v, 0], dt=dt, name='earth')
	jupiter = Particle(1.8986e27, [-7.7854e11,0,0],
		[-np.sin((dt/2)*(2*np.pi/jupiter_period))*jupiter_v,
		-np.cos((dt/2)*(2*np.pi/jupiter_period))*jupiter_v, 0], dt=dt, name='jupiter')

	return [sun, earth, jupiter]