
import random
import numpy as np

G = 6.67*10**-11

def normal_circle(mass: float, r: float, num: int) -> np.array:
    """
    Returns a list of num normally distributed positions within a circle of radius r.
    """
    positions = np.zeros((num,4))
    for i in range(num):
        distance = r * (random.random() ** 0.5)
        delta = 2 * np.pi * random.random()
        velocity = np.sqrt(G * mass * distance / r**2)
        positions[i] = [np.cos(delta) * distance, np.sin(delta) * distance, -np.sin(delta) * velocity, np.cos(delta) * velocity]
    return positions

def disc_exp(mass:float, Rh:float, num: int) -> np.array:
    """
    Returns a list of num distributed positions according to n = exp(-r/R) where r is the distance from the centre and R is the scale length.
    """
    positions = np.zeros((num,4))
    for i in range(num):

        ok_dist = False
        while not ok_dist:
            delta = 2 * np.pi * random.random()
            distance = - Rh * np.log(1 - random.uniform(0.2,1))
            x, y = (np.cos(delta) * distance, np.sin(delta) * distance)
            if all(np.sqrt((x-p[0])**2 + (y-p[1])**2) > 1e19 for p in positions):
                ok_dist = True

        velocity = np.sqrt(G * mass * (0.5 + 0.5 * (1 - np.exp(-distance/Rh))) / distance)
        positions[i] = [x, y, -np.sin(delta) * velocity, np.cos(delta) * velocity]

    return positions




