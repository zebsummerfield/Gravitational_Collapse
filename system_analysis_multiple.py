
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import numpy as np
from constants import *
from kolmogorov_smirnov import *

with open('system_q0', 'r') as f:
    stats_dict0 = json.load(f)
with open('system_q10', 'r') as f:
    stats_dict10 = json.load(f)
with open('system_q20', 'r') as f:
    stats_dict20 = json.load(f)
with open('system_q30', 'r') as f:
    stats_dict30 = json.load(f)

stats_dicts = [stats_dict0, stats_dict10, stats_dict20, stats_dict30]
Dalpha = 1.35810 / np.sqrt(stats_dict0['num_disc'])
print(Dalpha)
time_steps = np.linspace(0, len(stats_dict0['evolving_KS'])-1, len(stats_dict0['evolving_KS'])) * stats_dict0['time_step'] / (1000000*year_in_s)

mpl.rcParams["font.size"] = 18
fig, ax = plt.subplots(figsize=(10,8))

for i in range(4):
    radii = stats_dicts[i]['final_radii']
    CDF = []
    for r in radii:
        CDF.append(FS(radii, r))
    ax.plot(np.array(radii)/(1000*pc), np.array(CDF)*500, label=f'Q = {[0.0, 1.0, 2.0, 3.0][i]}', color=['darkorange', 'green', 'red', 'purple'][i])

distances = np.linspace(0, 5*scale_radius, num=1000)
lower = []
upper = []
for d in distances:
    lower.append(Lower(d, scale_radius, Dalpha))
    upper.append(Upper(d, scale_radius, Dalpha))
ax.fill_between(distances/(1000*pc), np.array(lower)*500, np.array(upper)*500, color='deepskyblue', alpha=0.25, label='Confidence Band')
ax.set_xlim(0,5*scale_radius/(1000*pc))
ax.set(xlabel='Distance From Galactic Centre [$kpc$]', ylabel='Cumulative Number of Particles')
plt.legend()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(21,7), constrained_layout = True)
for i in range(3):
    plot = axes[i].plot(np.array(stats_dicts[i]['final_pos'])[:,0]/(1000*pc), np.array(stats_dicts[i]['final_pos'])[:,1]/(1000*pc), marker='o', linestyle='none', markersize=1, color=['blue', 'green', 'red'][i])[0]
    axes[i].set(xlabel='[$kpc$]', ylabel='[$kpc$]')
    axes[i].title.set_text(f'Q = {[0.0, 1.0, 2.0][i]}')
    if i == 2:
        axes[i].set_xlim(-5*scale_radius/(1000*pc),5*scale_radius/(1000*pc))
        axes[i].set_ylim(-5*scale_radius/(1000*pc),5*scale_radius/(1000*pc))
    else:
        axes[i].set_xlim(-5*scale_radius/(1000*pc),5*scale_radius/(1000*pc))
        axes[i].set_ylim(-5*scale_radius/(1000*pc),5*scale_radius/(1000*pc))
plt.show()