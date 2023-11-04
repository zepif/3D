import numpy as np
from vector_functions import *
from settings import *

@njit(fastmath=True, cache=True)
def intersect_plane(ro, rd, parameters):
    # Ax + By + Cz + D = 0

    A, B, C, D = parameters
    
    P = (A, B, C)
    N = (0, 1, 0)
    denominator = dot_vecs3(rd, N)

    if abs(denominator) < 0.000001:
        return np.inf, (0,0,0), (0,0,0)
    
    t = dot_vecs3(sub_vecs3(P, ro), N) / denominator

    if t < 0:
        return np.inf, (0,0,0), (0,0,0)
    
    n = N
    p = sum_vecs3(ro, mul_vec3_n(rd, t))

    return t, p, n

@njit(fastmath=True, cache=True)
def intersect_sphere(ro, rd, parameters):
    # (x - Cx)^2 + (y - Cy)^2 + (z - Cz)^2 = R^2

    C, R = parameters[:-1], parameters[-1]

    a = dot_vecs3(rd, rd)
    OC = sub_vecs3(ro, C)
    b = 2 * dot_vecs3(rd, OC)
    c = dot_vecs3(OC, OC) - R * R
    disc = b * b - 4 * a * c

    if disc > 0:
        distSqrt = math.sqrt(disc)
        q = (-b - distSqrt) / 2.0 if b < 0 else (-b + distSqrt) / 2.0
        t0 = q / a
        t1 = c / q
        t0, t1 = min(t0, t1), max(t0, t1)

        if t1 >= 0:
            t = t1 if t0 < 0 else t0

        p = sum_vecs3(ro, mul_vec3_n(rd, t))
        n = normalize_vec3(sub_vecs3(p, C))

        return t, p, n
    
    return np.inf, (0,0,0), (0,0,0)

@njit(fastmath=True, cache=True)
def intersect_cube(ro, rd, parameters):
    # Cube defined by its center (Cx, Cy, Cz) and half side length L

    C, L = parameters[:-1], parameters[-1]

    t_near = -np.inf
    t_far = np.inf

    for i in range(3):
        if abs(rd[i]) < 1e-6:
            if ro[i] < C[i] - L or ro[i] > C[i] + L:
                return np.inf, (0, 0, 0), (0, 0, 0)
        else:
            t1 = (C[i] - L - ro[i]) / rd[i]
            t2 = (C[i] + L - ro[i]) / rd[i]

            if t1 > t2:
                t1, t2 = t2, t1

            t_near = max(t_near, t1)
            t_far = min(t_far, t2)

            if t_near > t_far:
                return np.inf, (0, 0, 0), (0, 0, 0)

    if t_near < 0:
        t = t_far
    else:
        t = t_near

    p = sum_vecs3(ro, mul_vec3_n(rd, t))

    if abs(p[0] - C[0]) > L - 1e-6:
        n = (1, 0, 0) if p[0] > C[0] else (-1, 0, 0)
    elif abs(p[1] - C[1]) > L - 1e-6:
        n = (0, 1, 0) if p[1] > C[1] else (0, -1, 0)
    else:
        n = (0, 0, 1) if p[2] > C[2] else (0, 0, -1)

    return t, p, n

@njit(fastmath=True, cache=True)
def intersect(ro, rd, obj):
    if obj[0] == 'sphere':
        return intersect_sphere(ro, rd, obj[1:])
    elif obj[0] == 'plane':
        return intersect_plane(ro, rd, obj[1:])
    elif obj[0] == 'cube':
        return intersect_cube(ro, rd, obj[1:])

@njit(fastmath=True, cache=True)
def get_color(ind, p):
    if ind == 0:
        return DARKGRAY if (int(p[0] * 2) % 2) == (int(p[2] * 2) % 2) else WHITE
    else:
        return (BLUE, RED, YELLOW, GREEN, ORANGE, PURPLE)[ind-1]