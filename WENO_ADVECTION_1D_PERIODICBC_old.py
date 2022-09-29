#!/usr/bin/python3
"""
@Author: Faranak Rajabi
@Description: Hamilton Jacobi WENO SOLVER FOR ADVECTION EQUATION in 1 dimension
    * phi_t + u*phi_x = 0
    * Forward euler in time
    * Periodic Boundary Condition
    * WENO scheme for finding the best stencill for ux  according to velocity
"""
import time
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

tic = time.time()
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
final_time_step = int(tf / dt)


# Defining the initial function phi0 as a square wave function
phi0 = []
num_xup = int((1 / 3 - min_x) / dx)
num_xlow = int((-1 / 3 - min_x) / dx)
for i in range(Nx):
    if num_xlow <= i <= num_xup:
        phi0.append(1)
    else:
        phi0.append(0)
phi0 = np.array(phi0)


def get_dphix(v1, v2, v3, v4, v5):
    """
    params v1, v2, v3, v4, and v5: our first-order approximations
    :return: The HJ WENO approximation of phi_x computed from {v1, v2, v3, v4, v5}
    """

    # Three possible choices
    p1 = v1 / 3 - 7 * v2 / 6 + 11 * v3 / 6
    p2 = -v2 / 6 + 5 * v3 / 6 + v4 / 3
    p3 = v3 / 3 + 5 * v4 / 6 - v5 / 6

    # Smoothness Coefficients for phix-
    s1 = (13. / 12) * ((v1 - 2 * v2 + v3) ** 2) + (1. / 4) * ((v1 - 4 * v2 + 3 * v3) ** 2)
    s2 = (13. / 12) * ((v2 - 2 * v3 + v4) ** 2) + (1. / 4) * ((v2 - v4) ** 2)
    s3 = (13. / 12) * ((v3 - 2 * v4 + v5) ** 2) + (1. / 4) * ((3 * v3 - 4 * v4 + v5) ** 2)

    # Alpha for ux-
    v = [v1, v2, v3, v4, v5]
    v = np.array(v)
    epsilon = 1e-6 * max(v ** 2) + 1e-99
    a1 = 0.1 / ((s1 + epsilon) ** 2)
    a2 = 0.6 / ((s2 + epsilon) ** 2)
    a3 = 0.3 / ((s3 + epsilon) ** 2)

    # Weight Coefficients for phix-
    w1 = a1 / (a1 + a2 + a3)
    w2 = a2 / (a1 + a2 + a3)
    w3 = a3 / (a1 + a2 + a3)

    return w1 * p1 + w2 * p2 + w3 * p3


# d(phi)/dx minus calculator
# This happens when the velocity is positive == if c1 > 0
def dphidx_minus_calculator(delta_x, current_phi):
    """
    :param delta_x: difference between two consecutive points on the grid
    :param current_phi: The current values for phi
    :return: The (approximated) derivatives of current_phi
    """

    x_grid_points = len(current_phi)  # The number of points on our grid

    def D_minus(x):
        """
        Function for first derivatives (D-) - Note that we have periodic BC
        :param x: position on our grid
        :return:
        """
        D_m = (current_phi[x % (x_grid_points - 1)] - current_phi[(x - 1) % (x_grid_points - 1)]) / delta_x
        return D_m

    dphidx = []
    # Stencils for phi_x minus
    for i in range(x_grid_points):

        v1 = D_minus(i - 2)
        v2 = D_minus(i - 1)
        v3 = D_minus(i)
        v4 = D_minus(i + 1)
        v5 = D_minus(i + 2)

        dphidx.append(get_dphix(v1, v2, v3, v4, v5))

    dphidx = np.array(dphidx)
    return dphidx


# d(phi)/dx plus calculator
# This happens when the velocity is negative == if c1<0
def dphidx_plus_calculator(delta_x, current_phi):
    """
    :param delta_x: difference between two consecutive points on the grid
    :param current_phi: The current values for phi
    :return: The (approximated) derivatives of current_phi
    """
    x_grid_points = len(current_phi)  # The number of points on our grid

    def D_plus(x):
        """
        Function for first derivatives (D+) - Note that we have periodic BC
        :param x: position on our grid
        :return:
        """
        D_p = (current_phi[(x + 1) % (x_grid_points - 1)] - current_phi[x % (x_grid_points - 1)]) / delta_x
        return D_p

    dphidx = []
    # Stencils for phi_x plus
    for i in range(x_grid_points):
        v1 = D_plus(i + 2)
        v2 = D_plus(i + 1)
        v3 = D_plus(i)
        v4 = D_plus(i - 1)
        v5 = D_plus(i - 2)

        dphidx.append(get_dphix(v1, v2, v3, v4, v5))

    dphidx = np.array(dphidx)
    return dphidx


# defining a function for calculating phi_np1 when c1 > 0
def phi_update(delta_x, phi_old, delta_t, c1):
    if c1 <= 0:
        # Velocity is negative, so we rely on Phi_x plus approximations
        diff = dphidx_plus_calculator(delta_x, phi_old)
    else:
        # Velocity is positive, so we rely on Phi_x minus approximations
        diff = dphidx_minus_calculator(delta_x, phi_old)

    phi_new = phi_old + delta_t * diff * (-c1)
    return phi_new


# Let's store the phi values for each time step, first we store the initial phi
phi_list = [phi0]

for n in range(0, final_time_step):
    # calculating the phi_n plus one by the phi_update function
    phi_np1 = phi_update(dx, phi_list[-1], dt, c1)
    phi_np2 = phi_update(dx, phi_np1, dt, c1)
    phi_np1half = .75 * phi_list[-1] + .25 * phi_np2
    phi_np3half = phi_update(dx, phi_np1half, dt, c1)
    phi_np1 = 1. / 3 * phi_list[-1] + 2. / 3 * phi_np3half

    phi_list.append(phi_np1)


# Here, we create what matplotlib calls an "artist"--in our case the axes ax.
fig, ax = plt.subplots()


def init():
    ax.clear()
    ax.set_xticks(np.linspace(0, Nx, 4), ["%.2f" % value for value in np.linspace(-1, 1, 4)])
    ax.set_xlim(0, Nx)
    ax.set_ylim(0, 1.2)
    return ax,  # Must return an iterable of artists.


def update(frame):

    data, time = frame

    init()

    ax.plot(range(0, Nx), phi_list[0], color='crimson', label='T=0s')
    ax.plot(range(0, Nx), data, color='steelblue', label='T=%.2fs' % time)
    ax.legend()

# plt.plot(range(0, Nx), phi_list[0], label="WENO Scheme for Advection equation at, time=" + str(tf))
# plt.show()

frames = []
for idx in range(len(phi_list)):
    frames.append([phi_list[idx], idx * dt])

ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=False, repeat=False)

# ani.save('animation.gif', writer='imagemagick', fps=60)
plt.show()

toc = time.time()
print(tic - toc)


