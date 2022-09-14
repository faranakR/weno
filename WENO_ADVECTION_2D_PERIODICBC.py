"""
@Author: Faranak Rajabi
@Description: Hamilton Jacobi WENO SOLVER FOR ADVECTION EQUATION in 2 dimensions
    * phi_t + c1*phi_x + c2*phi_y = 0
    * Periodic Boundary Condition
    * WENO scheme for finding the best stencill for ux  according to velocity
"""
import math
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from matplotlib.animation import FuncAnimation


def get_dphix(v1, v2, v3, v4, v5):
    """
    params v1, v2, v3, v4, and v5: our first-order approximations
    :return: The HJ WENO approximation of phi_x computed from {v1, v2, v3, v4, v5}
    this part can be used for the phi_y calculation
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
    epsilon = 1e-6 * np.max(v ** 2, axis=0) + 1e-99
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
    # phi_shift_minusx(i) = phi(i - x)
    phi_shift_minus1 = shift_vector_right(current_phi)
    phi_shift_minus2 = shift_vector_right(phi_shift_minus1)
    phi_shift_minus3 = shift_vector_right(phi_shift_minus2)

    # phi_shift_plusx(i) = phi(i + x)
    phi_shift_plus1 = shift_vector_left(current_phi)
    phi_shift_plus2 = shift_vector_left(phi_shift_plus1)

    v1 = (phi_shift_minus2 - phi_shift_minus3) / delta_x  # Phi(i-2) - Phi(i-3)
    v2 = (phi_shift_minus1 - phi_shift_minus2) / delta_x  # Phi(i-1) - Phi(i-2)
    v3 = (current_phi - phi_shift_minus1) / delta_x  # Phi(i) - Phi(i-1)
    v4 = (phi_shift_plus1 - current_phi) / delta_x  # Phi(i+1) - Phi(i)
    v5 = (phi_shift_plus2 - phi_shift_plus1) / delta_x  # Phi(i+2) - Phi(i+1)

    dphidx = get_dphix(v1, v2, v3, v4, v5)

    return dphidx


def shift_vector_right(vector):
    new_vector = np.zeros(vector.shape)
    new_vector[:, 0] = vector[:, -1]
    new_vector[:, 1:] = vector[:, :-1]
    return new_vector


def shift_vector_left(vector):
    new_vector = np.zeros(vector.shape)
    new_vector[:, -1] = vector[:, 0]
    new_vector[:, :-1] = vector[:, 1:]
    return new_vector


# d(phi)/dx plus calculator
# This happens when the velocity is negative == if c1<0
def dphidx_plus_calculator(delta_x, current_phi):
    """
    :param delta_x: difference between two consecutive points on the grid
    :param current_phi: The current values for phi
    :return: The (approximated) derivatives of current_phi
    """

    # phi_shift_minusx(i) = phi(i - x)
    phi_shift_minus1 = shift_vector_right(current_phi)
    phi_shift_minus2 = shift_vector_right(phi_shift_minus1)

    # phi_shift_plusx(i) = phi(i + x)
    phi_shift_plus1 = shift_vector_left(current_phi)
    phi_shift_plus2 = shift_vector_left(phi_shift_plus1)
    phi_shift_plus3 = shift_vector_left(phi_shift_plus2)

    v1 = (phi_shift_plus3 - phi_shift_plus2) / delta_x  # Phi(i+3) - Phi(i+2)
    v2 = (phi_shift_plus2 - phi_shift_plus1) / delta_x  # Phi(i+2) - Phi(i+1)
    v3 = (phi_shift_plus1 - current_phi) / delta_x  # Phi(i+1) - Phi(i)
    v4 = (current_phi - phi_shift_minus1) / delta_x  # Phi(i) - Phi(i-1)
    v5 = (phi_shift_minus1 - phi_shift_minus2) / delta_x  # Phi(i-1) - Phi(i-2)

    dphidx = get_dphix(v1, v2, v3, v4, v5)

    return dphidx


# defining a function for calculating phi_np1 for different types of u(velocity in x direction) and v(velocity in y
# direction)
def phi_update(delta_x, delta_y, phi_old, delta_t, u, v):
    if u <= 0 and v <= 0:
        # Velocity in x and y direction is negative, so we rely on Phi_x plus and Phi_y plus approximations
        diff_x = dphidx_plus_calculator(delta_x, phi_old)
        diff_y = dphidx_plus_calculator(delta_y, phi_old.transpose()).transpose()
    elif u <= 0 < v:
        # Velocity in x direction is negative and in y direction is positive, so we rely on Phi_x plus and Phi_y
        # minus approximations
        diff_x = dphidx_plus_calculator(delta_x, phi_old)
        diff_y = dphidx_minus_calculator(delta_y, phi_old.transpose()).transpose()
    elif v <= 0 < u:
        # Velocity in x direction is positive and in y direction is negative, so we rely on Phi_x minus and Phi_y
        # plus approximations
        diff_x = dphidx_minus_calculator(delta_x, phi_old)
        diff_y = dphidx_plus_calculator(delta_y, phi_old.transpose()).transpose()
    else:
        # Velocity in x and y direction is positive, so we rely on Phi_x minus and Phi_y minus approximations
        diff_x = dphidx_minus_calculator(delta_x, phi_old)
        diff_y = dphidx_minus_calculator(delta_y, phi_old.transpose()).transpose()

    phi_new = phi_old + delta_t * (diff_x * (-c1) + diff_y * (-c2))
    return phi_new


### For plotting
def init():
    fig.clear()
    ax_contour = fig.add_subplot(121)
    ax_phi = fig.add_subplot(122, projection='3d')
    ax_contour.set_aspect('equal', adjustable='box')
    for ax in [ax_phi, ax_contour]:
        ax.clear()
        ax.set_xticks(np.linspace(min_x, max_x, 4), ["%.2f" % value for value in np.linspace(min_x, max_x, 4)])
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_yticks(np.linspace(min_y, max_y, 4), ["%.2f" % value for value in np.linspace(min_y, max_y, 4)])
    ax_phi.set_zlim(-1, 0.2)
    # return ax,  # Must return an iterable of artists.
    ax_contour.set_aspect('equal', adjustable='box')

    return ax_phi, ax_contour


def update(frame):
    data, time = frame

    ax_phi, ax_contour = init()
    fig.suptitle('Time=%.2fs' % time)

    ax_phi.plot_surface(meshgrid_x, meshgrid_y, data, cmap=cm.coolwarm, linewidth=0.0)
    ax_phi.set_title('Solution for Advection Equation')

    ax_contour.contour(meshgrid_x, meshgrid_y, data, [0], colors=['crimson'])
    ax_contour.set_title('Zero Level-Set')

    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # ax.legend()


if __name__ == "__main__":
    parser = argparse.ArgumentParser('HJ WENO SOLVER for 2 dimensions')

    parser.add_argument('--c1', default=1 / math.sqrt(2), type=float, help='velocity in x direction')
    parser.add_argument('--c2', default=1 / math.sqrt(2), type=float, help='velocity in y direction')
    parser.add_argument('--min-x', default=-1, type=float)
    parser.add_argument('--max-x', default=1, type=float)
    parser.add_argument('--min-y', default=-1, type=float)
    parser.add_argument('--max-y', default=1, type=float)
    parser.add_argument('--x-grid-points-num', default=100, type=int, help='number of grid points on the x axis')
    parser.add_argument('--y-grid-points-num', default=100, type=int, help='number of grid points on the x axis')
    parser.add_argument('--CFL', default=0.9, type=float)
    parser.add_argument('--final-time', default=3., type=float)
    parser.add_argument('--output-file', default='animation.gif')

    params = parser.parse_args()

    c1 = params.c1
    min_x = params.min_x
    max_x = params.max_x
    D_size_x = max_x - min_x  # size of the spatial domain
    Nx = params.x_grid_points_num
    dx = D_size_x / Nx
    x_value = np.linspace(min_x, max_x, Nx)

    c2 = params.c2
    min_y = params.min_y
    max_y = params.max_y
    D_size_y = max_y - min_y  # size of the spatial domain
    Ny = params.y_grid_points_num
    dy = D_size_y / Ny
    y_value = np.linspace(min_y, max_y, Ny)

    tf = params.final_time  # final time
    dt = params.CFL / (abs(c1) / dx + abs(c2) / dy)
    final_time_step = int(tf / dt)

    tic = time.time()
    # Defining the initial function phi0 as a square wave function
    phi0 = np.zeros((Nx, Ny))
    xc = 0
    yc = 0
    r = 0.3
    for i in range(Nx):
        for j in range(Ny):
            phi0[j, i] = math.sqrt((x_value[i] - xc) ** 2 + (y_value[j] - yc) ** 2) - r

    # Let's store the phi values for each time step, first we store the initial phi
    phi_list = [phi0]

    for n in range(0, final_time_step):
        # calculating the phi_n plus one by the phi_update function
        phi_np1 = phi_update(dx, dy, phi_list[-1], dt, c1, c2)
        phi_np2 = phi_update(dx, dy, phi_np1, dt, c1, c2)
        phi_np1half = .75 * phi_list[-1] + .25 * phi_np2
        phi_np3half = phi_update(dx, dy, phi_np1half, dt, c1, c2)
        phi_np1 = 1. / 3 * phi_list[-1] + 2. / 3 * phi_np3half

        phi_list.append(phi_np1)

    meshgrid_x, meshgrid_y = np.meshgrid(x_value, y_value)

    # Here, we create what matplotlib calls an "artist"--in our case the axes ax.
    fig = plt.figure()

    frames = []
    for idx in range(len(phi_list)):
        frames.append([-phi_list[idx], idx * dt])

    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=False, repeat=False)
    toc = time.time()
    print(tic - toc)
    ani.save(params.output_file, writer='imagemagick', fps=60)
    plt.show()
