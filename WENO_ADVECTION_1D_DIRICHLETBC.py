import sys
import math
import numpy as np
import matplotlib.pyplot as plt

# HJ WENO SOLVER FOR ADVECTION EQUATION in 1D
# Advection equation in 1D: phi_t + u*phi_x =0
# Forward euler in time
# WENO scheme for finding the best stencill for ux  according to velocity
# Periodic BC
# ----------------------------------------------------------------------------
# 1D case
# velocity
c1 = 1  # velocity in x direction
# Spatial Domain
min_x = -1
max_x = 1
D_size = max_x - min_x  # size of the spatial domain
Nx = 200
dx = D_size / Nx
# CFL Condition
CFL = 0.9
# Time
tf = 3  # final time
dt = dx * CFL / abs(c1)


# ----------------------------------------------------------------------------
# Defining the initial function phi0 as a square wave function
def phi0(x_nodes, minimum_x, maximum_x):
    domain = maximum_x - minimum_x
    phi0 = []
    dx = domain / x_nodes

    up_num = math.floor((1 / 3 - min_x) / dx)
    low_num = math.floor((-1 / 3 - min_x) / dx)
    for i in range(x_nodes):
        if i <= low_num:
            phi0.append(0)
        elif low_num < i < up_num:
            phi0.append(1)
        else:
            phi0.append(0)
    return phi0


# ----------------------------------------------------------------------------
# initializing the problem
t = 0
phi_n = phi0(Nx, min_x, max_x)  # phi in nth time step (current time step)


# ----------------------------------------------------------------------------
# d(phi)/dx minus calculator
# This happens when the velocity is positive == if c1>0
def dphidx_minus_calculator(delta_x, x_grid_points, unknown_f):
    # Function for first derivatives (D-)
    # Three inputs: function, points, h
    def D_minus(array, x, h):
        D_m = (array[x] - array[x - 1]) / h
        return D_m

    # Stencils for phi_x minus
    for i in range(x_grid_points):
        if i == 0:
            v1 = D_minus(unknown_f, x_grid_points - 3, delta_x)
            v2 = D_minus(unknown_f, x_grid_points - 2, delta_x)
            v3 = D_minus(unknown_f, x_grid_points - 1, delta_x)
        elif i == 1:
            v1 = D_minus(unknown_f, x_grid_points - 2, delta_x)
            v2 = D_minus(unknown_f, x_grid_points - 1, delta_x)
        elif i == 2:
            v1 = D_minus(unknown_f, x_grid_points - 1, delta_x)
        elif i == x_grid_points - 1:
            v4 = D_minus(unknown_f, 1, delta_x)
            v5 = D_minus(unknown_f, 2, delta_x)
        elif i == x_grid_points - 2:
            v5 = D_minus(unknown_f, 1, delta_x)
        else:
            v1 = D_minus(unknown_f, i - 2, delta_x)
            v2 = D_minus(unknown_f, i - 1, delta_x)
            v3 = D_minus(unknown_f, i, delta_x)
            v4 = D_minus(unknown_f, i + 1, delta_x)
            v5 = D_minus(unknown_f, i + 2, delta_x)
    # ----------------------------------------------------------------------------
    # Three possible choices
    p1 = v1 / 3 - 7 * v2 / 6 + 11 * v3 / 6
    p2 = -v2 / 6 + 5 * v3 / 6 + v4 / 3
    p3 = v3 / 3 + 5 * v4 / 6 - v5 / 6
    # Smoothness Coefficients for phix-
    s1 = 13 / 12 * (v1 - 2 * v2 + v3) ** 2 + 1 / 4 * (v1 - 4 * v2 + 3 * v3) ** 2
    s2 = 13 / 12 * (v2 - 2 * v3 + v4) ** 2 + 1 / 4 * (v2 - v4) ** 2
    s3 = 13 / 12 * (v3 - 2 * v4 + v5) ** 2 + 1 / 4 * (3 * v3 - 4 * v4 + v5) ** 2
    # Alpha for ux-
    v = [v1, v2, v3, v4, v5]
    v = np.array(v)
    epsilon = 10 ** (-6) * max(v) + 10 ** (-99)
    a1 = 0.1 / (s1 + epsilon) ** 2
    a2 = 0.6 / (s2 + epsilon) ** 2
    a3 = 0.3 / (s3 + epsilon) ** 2
    # Weight Coefficients for phix-
    w1 = a1 / (a1 + a2 + a3)
    w2 = a2 / (a1 + a2 + a3)
    w3 = a3 / (a1 + a2 + a3)
    # Final
    dphidx = []
    for i in range(x_grid_points):
        dphidx.append(w1 * p1 + w2 * p2 + w3 * p3)  # problem with i-2, i-1, i+1, i+2

    dphidx = np.array(dphidx)
    return dphidx


# ----------------------------------------------------------------------------
# d(phi)/dx plus calculator
# This happens when the velocity is negative == if c1<0
def dphidx_plus_calculator(delta_x, x_grid_points, unknown_f):
    # Function for first derivatives (D+)
    # Three inputs: function, points, h
    def D_plus(array, x, h):
        D_p = (array[x + 1] - array[x]) / h
        return D_p

    # Stencils for phi_x plus
    for i in range(x_grid_points):
        if i == 0:
            v4 = D_plus(unknown_f, x_grid_points - 2, delta_x)
            v5 = D_plus(unknown_f, x_grid_points - 3, delta_x)
        elif i == 1:
            v5 = D_plus(unknown_f, x_grid_points - 2, delta_x)
        elif i == x_grid_points - 3:
            v1 = D_plus(unknown_f, 0, delta_x)
        elif i == x_grid_points - 2:
            v1 = D_plus(unknown_f, 1, delta_x)
            v2 = D_plus(unknown_f, 0, delta_x)
        elif i == x_grid_points - 1:
            v1 = D_plus(unknown_f, 2, delta_x)
            v2 = D_plus(unknown_f, 1, delta_x)
            v3 = D_plus(unknown_f, 0, delta_x)
        else:
            v1 = D_plus(unknown_f, i + 2, delta_x)
            v2 = D_plus(unknown_f, i + 1, delta_x)
            v3 = D_plus(unknown_f, i, delta_x)
            v4 = D_plus(unknown_f, i - 1, delta_x)
            v5 = D_plus(unknown_f, i - 2, delta_x)
    # ----------------------------------------------------------------------------
    # Three possible choices
    p1 = v1 / 3 - 7 * v2 / 6 + 11 * v3 / 6
    p2 = -v2 / 6 + 5 * v3 / 6 + v4 / 3
    p3 = v3 / 3 + 5 * v4 / 6 - v5 / 6
    # Smoothness Coefficients for phix-
    s1 = 13 / 12 * (v1 - 2 * v2 + v3) ** 2 + 1 / 4 * (v1 - 4 * v2 + 3 * v3) ** 2
    s2 = 13 / 12 * (v2 - 2 * v3 + v4) ** 2 + 1 / 4 * (v2 - v4) ** 2
    s3 = 13 / 12 * (v3 - 2 * v4 + v5) ** 2 + 1 / 4 * (3 * v3 - 4 * v4 + v5) ** 2
    # Alpha for ux-
    v = [v1, v2, v3, v4, v5]
    v = np.array(v)
    epsilon = 10 ** (-6) * max(v) + 10 ** (-99)
    a1 = 0.1 / (s1 + epsilon) ** 2
    a2 = 0.6 / (s2 + epsilon) ** 2
    a3 = 0.3 / (s3 + epsilon) ** 2
    # Weight Coefficients for phix-
    w1 = a1 / (a1 + a2 + a3)
    w2 = a2 / (a1 + a2 + a3)
    w3 = a3 / (a1 + a2 + a3)
    # Final
    dphidx = []
    for i in range(x_grid_points):
        dphidx.append(w1 * p1 + w2 * p2 + w3 * p3)  # problem with i-2, i-1, i+1, i+2
    return dphidx


# ----------------------------------------------------------------------------
if c1 > 0:
    for i in range(Nx):
        dif = dphidx_minus_calculator(dx, Nx, phi_n)
        phi_np1 = phi_n + dt * dif[i] * (-c1)
        phi_n = phi_np1
if c1 <= 0:
    for i in range(Nx):
        dif = dphidx_plus_calculator(dx, Nx, phi_n)
        phi_np1 = phi_n + dt * dif[i] * (-c1)
        phi_n = phi_np1
