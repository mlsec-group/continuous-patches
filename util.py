import numpy as np
import math

def bernstein_poly(i, n, t):
    return math.comb(n, i) * (t ** i) * ((1 - t) ** (n - i))

def bezier_curve(control_points, n_points=100):
    t = np.linspace(0, 1, n_points)
    n = len(control_points) - 1

    curve = np.zeros((n_points, 3))
    for i in range(n + 1):
        curve += np.outer(bernstein_poly(i, n, t), control_points[i])

    return curve