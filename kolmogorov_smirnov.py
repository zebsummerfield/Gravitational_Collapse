import numpy as np

def F0(r: float, Rh: float) -> float:
    return 1 - np.exp(-r/Rh)

def I(xi: float, x: float) -> int:
    if xi <= x:
        return 1
    else:
        return 0
    
def FS(positions: list, x: float) -> float:
    return (1/len(positions)) * sum([I(p, x) for p in positions])

def KS_test(positions: list, Rh: float) -> list:
    positions = np.sort(positions)
    statistics = []
    for index, xi in enumerate(positions):
        a = abs(FS(positions, xi) - F0(xi, Rh))
        if index == 0:
            b = F0(xi, Rh)
        else:
            b = abs(F0(xi, Rh) - FS(positions, positions[index-1]))
        statistics.append(max(a, b))
    return statistics

def Lower(r: float, Rh: float, Dalpha: float) -> float:
    return max(F0(r, Rh) - Dalpha, 0)

def Upper(r: float, Rh: float, Dalpha: float) -> float:
    return min(F0(r, Rh) + Dalpha, 1)

if __name__ == "__main__":
    from distributions import disc_all
    from constants import *
    from utils import modulus

    dt = 100000 * year_in_s
    num_disc = 500
    disc_particles = disc_all(disc_mass, scale_radius, halo_scale_radius, centre_mass, disc_density0, halo_density0, num_disc, dt)
    positions = [modulus(p.pos) for p in disc_particles]
    KS = KS_test(positions, scale_radius)
    print(max(KS))