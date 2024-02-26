
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import numpy as np
from constants import *
from kolmogorov_smirnov import *

with open('system_q1', 'r') as f:
    stats_dict = json.load(f)

Dalpha = 1.35810 / np.sqrt(stats_dict['num_disc'])
time_steps = np.linspace(0, len(stats_dict['evolving_KS'])-1, len(stats_dict['evolving_KS'])) * stats_dict['time_step'] / (1000000*year_in_s)
radii = stats_dict['final_radii']
exp = []
lower = []
upper = []
for r in radii:
    exp.append(F0(r, scale_radius))
    lower.append(Lower(radii, r, Dalpha))
    upper.append(Upper(radii, r, Dalpha))

mpl.rcParams["font.size"] = 20
fig, ax = plt.subplots(figsize=(10,10))
ax.plot(radii, exp, label='exponential cumulative distribution')
ax.plot(radii, lower, label='lower confidence band')
ax.plot(radii, upper, label='upper confidence band')
plt.legend()
plt.show()