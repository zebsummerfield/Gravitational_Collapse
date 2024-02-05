import numpy as np
from particle import Particle
import distributions
from utils import *
import time
from barneshut import Node
from multiprocessing import Pool
import matplotlib.pyplot as plt
import matplotlib as mpl
import json

nums = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
with open('comparison_data', 'r') as f:
    l = json.load(f)


mpl.rcParams["font.size"] = 20
fig, ax = plt.subplots(figsize=(10,10))
ax.set(xlabel='Number of Partices', ylabel='Time required to permute the system one timestep [s]')
plt.gcf().set_tight_layout(True) # To prevent the xlabel being cut off
ax.set_xscale('log')
ax.set_xlim(100,20000)
ax.set_ylim([0, 100])
ax.plot(nums, l[0], label='Brute Force Serial')
ax.plot(nums, l[1], label='Brute Force Parallel')
ax.plot(nums, l[2], label='Barnes Hut Serial')
ax.plot(nums, l[3], label='Barnes Hut Parallel')
plt.legend()
plt.show()

                    