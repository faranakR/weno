import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

u = 1
Nx = 200
xmin = -1
xmax = 1
dx = (xmax - xmin) / Nx
dt = 0.9 * dx
tf = 3

u0 = []
# num_xlow = int((-1 / 3 - xmin) / dx)
num_xup = int((1 / 3 - xmin) / dx)
num_xlow = int((-1 / 3 - xmin) / dx)
for i in range(Nx):
    if num_xlow <= i <= num_xup:
        u0.append(1)
    else:
        u0.append(0)


# plt.plot(range(0, Nx), u0)
# plt.show()


# def ux_1_minus(dx, u0, index):
#     dif = (u0[index] - u0[index - 1]) / dx
#     return dif
#
#
# ux_list = []
# for i in range(Nx):
#     ux_list.append(ux_1_minus(dx, u0, i))


def u_update(u_old):
    u_new = []
    for i in range(0, Nx):
        if i == 0:
            u_new_i = u_old[Nx - 1] + dt * (-u) * (u_old[Nx - 1] - u_old[Nx - 2]) / dx
        else:
            u_new_i = u_old[i] + dt * (-u) * (u_old[i] - u_old[i - 1]) / dx
        u_new.append(u_new_i)
    return u_new


final_ts = int(tf / dt)
print(final_ts)
u_list = [u0]
for n in range(final_ts):
    u_np1 = u_update(u_list[-1])
    u_list.append(u_np1)

# Here, we create what matplotlib calls an "artist"--in our case the axes ax.
fig, ax = plt.subplots()


def init():
    ax.clear()
    ax.set_xticks(np.linspace(0, Nx, 4), ["%.2f" % value for value in np.linspace(-1, 1, 4)])
    ax.set_xlim(0, Nx)
    ax.set_ylim(0, 1.2)
    return ax,  # Must return an iterable of artists.


def update(frame):
    init()
    ax.plot(range(0, Nx), frame)


ani = FuncAnimation(fig, update, frames=u_list, init_func=init, blit=False, repeat=False)

plt.show()