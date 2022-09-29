"""
@Author: Faranak Rajabi
@Description: Hamilton Jacobi ENO SOLVER FOR ADVECTION EQUATION in 1 dimension
    * phi_t + u*phi_x = 0
    * Forward euler in time
    * Periodic Boundary Condition
    * ENO scheme for finding the best stencill for ux  according to velocity
"""
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation


# def get_dphix(v1, v2, v3, v4, v5):
#     """
#     params v1, v2, v3, v4, and v5: our first-order approximations
#     :return: The HJ WENO approximation of phi_x computed from {v1, v2, v3, v4, v5}
#     """
#
#     # Three possible choices
#     p1 = v1 / 3 - 7 * v2 / 6 + 11 * v3 / 6
#     p2 = -v2 / 6 + 5 * v3 / 6 + v4 / 3
#     p3 = v3 / 3 + 5 * v4 / 6 - v5 / 6
#
#     # Smoothness Coefficients for phix-
#     s1 = (13. / 12) * ((v1 - 2 * v2 + v3) ** 2) + (1. / 4) * ((v1 - 4 * v2 + 3 * v3) ** 2)
#     s2 = (13. / 12) * ((v2 - 2 * v3 + v4) ** 2) + (1. / 4) * ((v2 - v4) ** 2)
#     s3 = (13. / 12) * ((v3 - 2 * v4 + v5) ** 2) + (1. / 4) * ((3 * v3 - 4 * v4 + v5) ** 2)
#
#     # Alpha for ux-
#     v = [v1, v2, v3, v4, v5]
#     v = np.array(v)
#     epsilon = 1e-6 * np.max(v ** 2, axis=0) + 1e-99
#     a1 = 0.1 / ((s1 + epsilon) ** 2)
#     a2 = 0.6 / ((s2 + epsilon) ** 2)
#     a3 = 0.3 / ((s3 + epsilon) ** 2)
#
#     # Weight Coefficients for phix-
#     w1 = a1 / (a1 + a2 + a3)
#     w2 = a2 / (a1 + a2 + a3)
#     w3 = a3 / (a1 + a2 + a3)
#
#     return w1 * p1 + w2 * p2 + w3 * p3
#
#
# # d(phi)/dx minus calculator
# # This happens when the velocity is positive == if c1 > 0
# def dphidx_minus_calculator(delta_x, current_phi):
#     """
#     :param delta_x: difference between two consecutive points on the grid
#     :param current_phi: The current values for phi
#     :return: The (approximated) derivatives of current_phi
#     """
#     # phi_shift_minusx(i) = phi(i - x)
#     phi_shift_minus1 = shift_vector_right(current_phi)
#     phi_shift_minus2 = shift_vector_right(phi_shift_minus1)
#     phi_shift_minus3 = shift_vector_right(phi_shift_minus2)
#
#     # phi_shift_plusx(i) = phi(i + x)
#     phi_shift_plus1 = shift_vector_left(current_phi)
#     phi_shift_plus2 = shift_vector_left(phi_shift_plus1)
#
#     v1 = (phi_shift_minus2 - phi_shift_minus3) / delta_x  # Phi(i-2) - Phi(i-3)
#     v2 = (phi_shift_minus1 - phi_shift_minus2) / delta_x  # Phi(i-1) - Phi(i-2)
#     v3 = (current_phi - phi_shift_minus1) / delta_x  # Phi(i) - Phi(i-1)
#     v4 = (phi_shift_plus1 - current_phi) / delta_x  # Phi(i+1) - Phi(i)
#     v5 = (phi_shift_plus2 - phi_shift_plus1) / delta_x  # Phi(i+2) - Phi(i+1)
#
#     dphidx = get_dphix(v1, v2, v3, v4, v5)
#
#     return dphidx
#
#
def shift_vector_right(vector):
    new_vector = np.zeros(vector.shape)
    new_vector[0] = vector[-1]
    new_vector[1:] = vector[:-1]
    return new_vector


def shift_vector_left(vector):
    new_vector = np.zeros(vector.shape)
    new_vector[-1] = vector[0]
    new_vector[:-1] = vector[1:]
    return new_vector


#
#
# # d(phi)/dx plus calculator
# # This happens when the velocity is negative == if c1<0
# def dphidx_plus_calculator(delta_x, current_phi):
#     """
#     :param delta_x: difference between two consecutive points on the grid
#     :param current_phi: The current values for phi
#     :return: The (approximated) derivatives of current_phi
#     """
#
#     # phi_shift_minusx(i) = phi(i - x)
#     phi_shift_minus1 = shift_vector_right(current_phi)
#     phi_shift_minus2 = shift_vector_right(phi_shift_minus1)
#
#     # phi_shift_plusx(i) = phi(i + x)
#     phi_shift_plus1 = shift_vector_left(current_phi)
#     phi_shift_plus2 = shift_vector_left(phi_shift_plus1)
#     phi_shift_plus3 = shift_vector_left(phi_shift_plus2)
#
#     v1 = (phi_shift_plus3 - phi_shift_plus2) / delta_x  # Phi(i+3) - Phi(i+2)
#     v2 = (phi_shift_plus2 - phi_shift_plus1) / delta_x  # Phi(i+2) - Phi(i+1)
#     v3 = (phi_shift_plus1 - current_phi) / delta_x  # Phi(i+1) - Phi(i)
#     v4 = (current_phi - phi_shift_minus1) / delta_x  # Phi(i) - Phi(i-1)
#     v5 = (phi_shift_minus1 - phi_shift_minus2) / delta_x  # Phi(i-1) - Phi(i-2)
#
#     dphidx = get_dphix(v1, v2, v3, v4, v5)
#
#     return dphidx


def compute_diff_table(phi_old, order=3):
    diff_table = np.zeros((phi_old.shape[0], order + 1))
    diff_table[:, 0] = phi_old

    for dim in range(1, order + 1):
        diff_table[:, dim] = (diff_table[:, dim - 1] - shift_vector_right(diff_table[:, dim - 1])) / (dim * dx)
    # For order 3:
    # diff_table[i, 1] = D^1_{i-1/2}
    # diff_table[i, 2] = D^2_{i-1}
    # diff_table[i, 3] = D^3_{i-3/2}

    return diff_table


def phi_update(delta_x, phi_old, dt, c1):
    diff_list = compute_diff_table(phi_old, 3)
    q1_prime = diff_list[:, 1]
    q2_prime = diff_list[:, 2]
    q3_prime = diff_list[:, 3]
    phi_x_poly = np.zeros((np.shape(phi_old)))
    for i in range(phi_old.shape[0]):
        if c1 > 0:
            k = i - 1
        else:
            k = i

        phi_x_poly[i] = q1_prime[(k+1) % phi_old.shape[0]]

        d2_k = q2_prime[(k + 1) % phi_old.shape[0]]
        d2_kplus1 = q2_prime[(k + 2) % phi_old.shape[0]]
        if abs(d2_k) <= abs(d2_kplus1):
            phi_x_poly[i] += d2_k * (2 * (i - k) - 1) * delta_x  # q_prime2 = c * 2 * ((i - k) - 1) * dx
            # and c = D^2_(i - 1)
            k_star = k - 1
        else:
            phi_x_poly[i] += d2_kplus1 * (2 * (i - k) - 1) * delta_x  # q_prime2 = c * 2 * ((i - k) - 1) *
            # dx, in this case c = D^2_(i)
            k_star = k

        if abs(q3_prime[(k_star+2) % phi_old.shape[0]]) <= abs(q3_prime[(k_star + 3) % phi_old.shape[0]]):
            phi_x_poly[i] += q3_prime[(k_star+2) % phi_old.shape[0]] * (3 * (i - k_star) ** 2 - 6 * (i - k_star) + 2) * (delta_x ** 2)  # c* * (3(i - k*)2 - 6(i -
            # k*) + 2) * (dx)^2, in this case k* = k - 1 = i - 2, and c = D^3_(i - 3/2)
        else:
            phi_x_poly[i] += q3_prime[(k_star + 3) % phi_old.shape[0]] * (3 * (i - k_star) ** 2 - 6 * (i - k_star) + 2) * (delta_x ** 2)  # c* * (3(i - k*)2 - 6(i
            # - k*) + 2) * (dx)^2, in this case k* = k - 1 = i - 1, and c = D^3_(i - 1/2)

    phi_new = dt * phi_x_poly * (-c1) + phi_old
    return phi_new


# # defining a function for calculating phi_np1 when c1 > 0
# def phi_update(delta_x, phi_old, delta_t, c1):
#     diff_table = compute_diff_table(phi_old)
#     if c1 > 0:
#
#     else:
#
#     return phi_new


### For plotting
def init():
    ax.clear()
    ax.set_xticks(np.linspace(0, Nx, 4), ["%.2f" % value for value in np.linspace(min_x, max_x, 4)])
    ax.set_xlim(0, Nx)
    ax.set_ylim(0, 1.2)
    return ax,  # Must return an iterable of artists.


def update(frame):
    data, time = frame

    init()

    ax.plot(range(0, Nx), phi_list[0], color='crimson', label='T=0s')
    ax.plot(range(0, Nx), data, color='steelblue', label='T=%.2fs' % time)
    ax.legend()


if __name__ == "__main__":
    parser = argparse.ArgumentParser('HJ WENO SOLVER for 1 dimension')

    parser.add_argument('--c1', default=1, type=float, help='velocity in x direction')
    parser.add_argument('--min-x', default=-1, type=float)
    parser.add_argument('--max-x', default=1, type=float)
    parser.add_argument('--grid-points-num', default=200, type=int, help='number of grid points on the x axis')
    parser.add_argument('--CFL', default=0.9, type=float)
    parser.add_argument('--final-time', default=3., type=float)
    parser.add_argument('--output-file', default='animation.gif')

    params = parser.parse_args()

    c1 = params.c1
    min_x = params.min_x
    max_x = params.max_x
    D_size = max_x - min_x  # size of the spatial domain
    Nx = params.grid_points_num
    dx = D_size / Nx

    tf = params.final_time  # final time
    dt = dx * params.CFL / abs(c1)
    final_time_step = int(tf / dt)

    tic = time.time()
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

    frames = []
    for idx in range(len(phi_list)):
        frames.append([phi_list[idx], idx * dt])

    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=False, repeat=False)
    toc = time.time()
    print(toc - tic)
    # ani.save(params.output_file, writer='imagemagick', fps=60)
    plt.show()