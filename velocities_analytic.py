import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt

G = 6.67*10**-11
epsilon = 2e19

def gen_diff_potential(r: float, density0: float, Rh: float, mass_c: float) -> float:
    y = r / (2 * Rh)
    return (4 * np.pi * G * density0 * Rh * y**2 *(sp.i0(y) * sp.k0(y) - sp.i1(y) * sp.k1(y)) + G * r * mass_c / (r**2 + epsilon**2)) / r

def gen_v(r: float, density0: float, Rh: float) -> float:
    y = r / (2 * Rh)
    return np.sqrt((4 * np.pi * G * density0 * Rh * y**2 *(sp.i0(y) * sp.k0(y) - sp.i1(y) * sp.k1(y))))

def gen_v_wcm(r: float, density0: float, Rh: float, mass_c: float) -> float:
    y = r / (2 * Rh)
    return np.sqrt((4 * np.pi * G * density0 * Rh * y**2 *(sp.i0(y) * sp.k0(y) - sp.i1(y) * sp.k1(y))) + G * r * mass_c / (r**2 + epsilon**2))
    
def gen_angular_v(r: float, density0: float, Rh: float, mass_c: float) -> float:
    return gen_v_wcm(r, density0, Rh, mass_c) / r

def gen_diff_diff_potential(r: float, density0: float, Rh: float, mass_c: float, dx=1e16) -> float:
    return (gen_diff_potential(r + dx, density0, Rh, mass_c) - gen_diff_potential(r - dx, density0, Rh, mass_c)) / (2*dx)

def gen_epicyclic(r: float, density0: float, Rh: float, mass_c: float) -> float:
    term1 = 3 * gen_v_wcm(r , density0, Rh, mass_c)**2 / r**2
    term2 = gen_diff_diff_potential(r , density0, Rh, mass_c)
    return np.sqrt(term1 + term2)

def gen_density(r: float, density0: float, Rh: float) -> float:
    return density0 * np.exp(-r/Rh)

def gen_v_dispersion_radial(r: float, density0: float, Rh: float, mass_c: float, h=1.2) -> float:
    return h * 3.36 * G * gen_density(r, density0, Rh) / gen_epicyclic(r, density0, Rh, mass_c)

def gen_v_dispersion_azimuthal(r: float, density0: float, Rh: float, mass_c: float, h=1.2) -> float:
    return gen_v_dispersion_radial(r, density0, Rh, mass_c, h=h) * gen_epicyclic(r, density0, Rh, mass_c) / (2 * gen_angular_v(r, density0, Rh, mass_c))



if __name__ =='__main__':

    solar_mass = 2e30
    mass = 1e11 * solar_mass
    pc = 3.086e16
    Rh = 3 * 1000 * pc
    density0 = mass / (2 * np.pi * Rh**2)

    distances = np.linspace(0.02, 10*Rh, num=1000)
    velocities = []
    velocities2 = []
    epicycles = []
    v_dispersions = []
    densities = []
    for distance in distances:
        velocities.append(gen_v(distance, density0, Rh))
        velocities2.append(gen_v_wcm(distance, density0, Rh, 0.2*mass))
        epicycles.append(gen_epicyclic(distance, density0, Rh, 0.2*mass))
        v_dispersions.append(gen_v_dispersion_radial(distance, density0, Rh, 0.2*mass))
        densities.append(gen_density(distance, density0, Rh))
        
    fig, ax = plt.subplots(figsize=(10,10))
    ax.plot(distances, v_dispersions)
    ax.plot(distances, velocities2)
    plt.show()
