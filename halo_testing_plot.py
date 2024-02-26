"""
Generates a halo of dark matter particles distributed according to an NFW.
"""

import numpy as np
import matplotlib.pyplot as plt
import distributions
import matplotlib as mpl
from constants import *
from utils import modulus

num = 10000
year_in_s = 365 * 24 * 3600
dt = 100000 * year_in_s

positions = np.array(distributions.halo_NFW_positions(halo_scale_radius, virial_radius, num))

mpl.rcParams["font.size"] = 15
fig, axes = plt.subplots(1, 2, figsize=(20,10))

axes[0].remove()
axes[0] = fig.add_subplot(1,2,1, projection='3d')
axes[0].plot3D(positions[:,0]/(1000*pc), positions[:,1]/(1000*pc), positions[:,2]/(1000*pc), marker='o', linestyle='none', markersize=1.5)
axes[0].set_xlabel('[$kpc$]')
axes[0].set_ylabel('[$kpc$]')
axes[0].set_zlabel('[$kpc$]')
rmax = virial_radius/(1000*pc)
axes[0].set_xlim(-rmax, rmax)
axes[0].set_ylim(-rmax, rmax)
axes[0].set_zlim(-rmax, rmax)

radii = []
for p in positions:
    radii.append(modulus(p))
axes[1].hist(radii, bins=100)
plt.show()



