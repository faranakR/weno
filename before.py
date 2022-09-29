import sys
import math
import matplotlib.pyplot as plt

# generating mesh
a = 1  # advection coefficient
omega = 2
nodesNum = 200
dx = omega / (nodesNum - 1)
dt = 0.5 * dx
t_final = 1
final_time_step = int(t_final / dt)

# smooth initial function
# U0 = []
# for i in range(0, nodesNum):
#     U0.append(math.sin(2 * math.pi * i * dx))
#
# plt.plot(range(0, nodesNum), U0, label="initial function, t = 0")
# ----------------------------------------------------------------------------------------------------------------------
# non-smooth initial function
U0 = []
n_up = int((1 + 1/3) / dx)
n_low = int((1 + -1/3) / dx)
for i in range(0, nodesNum):
    if n_low <= i <= n_up:
        U0.append(1)
    else:
        U0.append(0)

plt.plot(range(0, nodesNum), U0, label="non-smooth initial function, t = 0")
# ----------------------------------------------------------------------------------------------------------------------
# exact solution for smooth function
# U_exact = []
# for i in range(0, nodesNum):
#     U_exact.append(math.sin(2 * math.pi * (i * dx - t_final)))
#
# plt.plot(range(0, nodesNum), U_exact, label="exact solution at t =" + str(t_final))
# ----------------------------------------------------------------------------------------------------------------------
# exact solution for non-smooth function
U_exact = []
n1 = int((2/3) / dx)
n2 = int(1 / dx)
for i in range(0, nodesNum):
    if n1 <= i <= n2:
        U_exact.append(1)
    else:
        U_exact.append(0)

plt.plot(range(0, nodesNum), U_exact, label="exact solution at t =" + str(t_final))
# ----------------------------------------------------------------------------------------------------------------------
# implementing Upwind method
# since a = 1, the wave propagates to the right direction :: using ui and ui-1

def update_u(U_old):

    CFL = -a * dx * dt
    U_new = []

    for i in range(0, nodesNum):

        if i == 0:
            U_new_i = -CFL * U_old[nodesNum - 2] + (1 + CFL) * U_old[i]
        else:
            U_new_i = -CFL * U_old[i-1] + (1 + CFL) * U_old[i]

        U_new.append(U_new_i)

    return U_new


# time steps & updating
U_list = [U0]
for n in range(0, final_time_step):
    U_new = update_u(U_list[-1])
    U_list.append(U_new)

# plotting
print(dx)
print(dt)
print(final_time_step)

plt.plot(range(0, nodesNum), U_list[-1], label="Upwind_scheme, t =" + str(t_final))
# ----------------------------------------------------------------------------------------------------------------------
# implementing Lax method

def update_u(U_old):
    alpha = -dt / (2 * dx)
    beta = dt ** 2 / (2 * dx ** 2)
    U_new = []

    for i in range(0, nodesNum):

        if i == 0:
            U_new_i = alpha * (U_old[i + 1] - U_old[nodesNum - 2]) + beta * (U_old[i + 1] + U_old[nodesNum - 2]) + (
                        1 - 2 * beta) * U_old[i]
        elif i == nodesNum - 1:
            U_new_i = alpha * (U_old[1] - U_old[i - 1]) + beta * (U_old[1] + U_old[nodesNum - 2]) + (1 - 2 * beta) * \
                      U_old[i]
        else:
            U_new_i = alpha * (U_old[i + 1] - U_old[i - 1]) + beta * (U_old[i + 1] + U_old[i - 1]) + (1 - 2 * beta) * \
                      U_old[i]

        U_new.append(U_new_i)

    return U_new


# time steps & updating
U_list = [U0]
for n in range(0, final_time_step):
    U_new = update_u(U_list[-1])
    U_list.append(U_new)

# plotting
plt.plot(range(0, nodesNum), U_list[-1], label="Lax_Wendorrf, t =" + str(t_final))
# ----------------------------------------------------------------------------------------------------------------------
# implementing Beam-warming method

def update_u(U_old):
    alpha = -dt / (2 * dx)
    beta = dt ** 2 / (2 * dx ** 2)
    U_new = []

    for i in range(0, nodesNum):

        if i == 0:
            U_new_i = (3 * alpha + beta + 1) * U_old[i] + (-4 * alpha - 2 * beta) * U_old[i-1] + (alpha + beta) * U_old[nodesNum - 3]
        else:
            U_new_i = (3 * alpha + beta + 1) * U_old[i] + (-4 * alpha - 2 * beta) * U_old[i-1] + (alpha + beta) * U_old[i-2]

        U_new.append(U_new_i)

    return U_new


# time steps & updating
U_list = [U0]
for n in range(0, final_time_step):
    U_new = update_u(U_list[-1])
    U_list.append(U_new)

# plotting
print(dx)
print(dt)
print(final_time_step)

plt.plot(range(0, nodesNum), U_list[-1], label="Beam Warming, t =" + str(t_final))
plt.legend()
plt.show()
