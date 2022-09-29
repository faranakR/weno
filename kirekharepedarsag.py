import sys
import numpy as np
import math
import matplotlib.pyplot as plt

x_min = -1
x_max = 1
Domain_size = x_max - x_min
Nx = 200
dx = Domain_size / Nx
c1 = 1
t_final = 3
CFL = 0.8
dt = dx * CFL / abs(c1)


# ______________________________________________________________________________________________________________________
# Defining the initial square wave
def phi0(x_nodes, minimum_x, maximum_x):
    domain = maximum_x - minimum_x
    phi0 = []
    dx = domain / x_nodes
    up_num = math.floor((1 / 3 - x_min) / dx)
    low_num = math.floor((-1 / 3 - x_min) / dx)
    for i in range(x_nodes):
        if i <= low_num:
            phi0.append(0)
        elif low_num < i < up_num:
            phi0.append(1)
        else:
            phi0.append(0)
    return phi0


# ______________________________________________________________________________________________________________________
def dif(j, h, u):
    dudx = (u[j] - u[j - 1]) / h
    return dudx
phi_n = phi0(Nx, x_min, x_max)
print(dif(45, dx, phi_n))


# ______________________________________________________________________________________________________________________
def phi_update(nodes_number, phi_old):  # remember the output is a list
    phi_new = []
    for i in range(nodes_number):
        if i == 0:
            kir = nodes_number - 1
            phi_new_i = phi_old[i] + (-c1) * dt * dif(kir, dx, phi_old)
        else:
            phi_new_i = phi_old[i] + (-c1) * dt * dif(i, dx, phi_old)
        phi_new = phi_new.append(phi_new_i)
        return phi_new


# ______________________________________________________________________________________________________________________
final_time_step = int(t_final / dt)
phi_n = phi0(Nx, x_min, x_max)
list_Of_Phi_np1 = [phi0(Nx, x_min, x_max)]
for j in range(0, final_time_step):
    phi_np1 = phi_update(Nx, phi_n)
    list_Of_Phi_np1.append(phi_np1)
    phi_n = phi_np1
print(list_Of_Phi_np1[-1])