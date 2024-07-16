
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import numpy as np
from constants import *
from kolmogorov_smirnov import *

with open('system_q40', 'r') as f:
    stats_dict = json.load(f)

Dalpha = 1.35810 / np.sqrt(stats_dict['num_disc'])
print(Dalpha)
time_steps = np.linspace(0, len(stats_dict['evolving_KS'])-1, len(stats_dict['evolving_KS'])) * stats_dict['time_step'] / (1000000*year_in_s)
radii = stats_dict['final_radii']
print(max(stats_dict['final_KS']))
CDF = []
lower = []
upper = []
for r in radii:
    CDF.append(FS(radii, r))
    lower.append(Lower(r, scale_radius, Dalpha))
    upper.append(Upper(r, scale_radius, Dalpha))

# mpl.rcParams["font.size"] = 20
# fig, ax = plt.subplots(figsize=(10,10))
# ax.plot(radii, CDF, label='cumulative distribution')
# ax.fill_between(radii, lower, upper, color='b', alpha=0.1, label='confidence band')
# plt.legend()
# plt.show()

# fig, ax = plt.subplots(figsize=(10,10))
# ax.plot(time_steps, stats_dict['evolving_KS'], label='KS')
# plt.legend()
# plt.show()

fig, ax = plt.subplots(figsize=(10,10))
plot = ax.plot(np.array(stats_dict['start_pos'])[:,0]/(1000*pc), np.array(stats_dict['start_pos'])[:,1]/(1000*pc), marker='o', linestyle='none', markersize=2)[0]
ax.set(xlabel='[$kpc$]', ylabel='[$kpc$]')
ax.set_xlim(-5*scale_radius/(1000*pc),5*scale_radius/(1000*pc))
ax.set_ylim(-5*scale_radius/(1000*pc),5*scale_radius/(1000*pc))
plt.axis('off')
plt.show()

# fig, ax = plt.subplots(figsize=(10,10))
# plot = ax.plot(np.array(stats_dict['final_pos'])[:,0]/(1000*pc), np.array(stats_dict['final_pos'])[:,1]/(1000*pc), marker='o', linestyle='none', markersize=1, color='red')[0]
# ax.set(xlabel='[$kpc$]', ylabel='[$kpc$]')
# ax.set_xlim(-5*scale_radius/(1000*pc),5*scale_radius/(1000*pc))
# ax.set_ylim(-5*scale_radius/(1000*pc),5*scale_radius/(1000*pc))
# plt.show()