# A simple centralized multi-agent RMP example: 
#            exchange positions while avoiding collisions
# @author Anqi Li
# @date April 8, 2019

from rmp import RMPRoot, RMPNode
from rmp_leaf import CollisionAvoidanceCentralized, GoalAttractorUni

import numpy as np
from numpy.linalg import norm
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

import math
from scipy.spatial.distance import pdist

# ---------------------------------------------
# build the rmp tree


N = 10
theta = np.arange(0, 2 * np.pi, 2 * np.pi / N)
x_g = np.stack((np.cos(theta), np.sin(theta))).T * 10

r = RMPRoot('root')


def create_mappings_robot(i):
    phi = lambda y, i=i: np.array([[y[2 * i, 0]], [y[2 * i + 1, 0]]])
    J = lambda y, i=i: np.concatenate((
            np.zeros((2, 2 * i)),
            np.eye(2),
            np.zeros((2, 2 * (N - i - 1)))), axis=1)
    J_dot = lambda y, y_dot: np.zeros((2, 2 * N))

    return phi, J, J_dot



def create_mappings_pair(i, j):
    assert i < j
    phi = lambda y, i=i, j=j: np.array([[y[2 * i, 0]], [y[2 * i + 1, 0]], [y[2 * j, 0]], [y[2 * j + 1, 0]]])
    J = lambda y, i=i, j=j: np.concatenate(
                (np.concatenate((
                    np.zeros((2, 2 * i)),
                    np.eye(2),
                    np.zeros((2, 2 * (N - i - 1)))), axis=1),
                np.concatenate((
                    np.zeros((2, 2 * j)),
                    np.eye(2),
                    np.zeros((2, 2 * (N - j - 1)))), axis=1)),
                axis=0)
    J_dot = lambda y, y_dot: np.zeros((4, 2 * N))

    return phi, J, J_dot


robots = []
for i in range(N):
    phi, J, J_dot = create_mappings_robot(i)
    robot = RMPNode('robot_' + str(i), r, phi, J, J_dot)
    robots.append(robot)

count = 0
pairs = []
for i in range(N):
    for j in range(N):
        if i >= j:
            continue
        phi, J, J_dot = create_mappings_pair(i, j)
        pair = RMPNode('pair_' + str(i) + '_' + str(j), r, phi, J, J_dot)
        pairs.append(pair)
        count += 1

gas = []
for i in range(N):
    ga = GoalAttractorUni(
        'ga_robot_' + str(i),
        robots[i],
        x_g[i],
        alpha = 1,
        gain = 1,
        eta = 2)
    gas.append(ga)

iacas = []
for i in range(count):
    iaca = CollisionAvoidanceCentralized(
            'ca_' + pairs[i].name,
            pairs[i],
            R = 1)
    iacas.append(iaca)
# ----------------------------------------------

# -----------------------------------------
# possible initial configurations
x_0 = - x_g + np.random.randn(*x_g.shape) * 0.2

x = x_0.reshape(-1)
x_dot = np.zeros_like(x)

state_0 = np.concatenate((x, x_dot), axis=None)


r.set_root_state(x, x_dot)
r.pushforward()
r.pullback()
# --------------------------------------------

# --------------------------------------------
# dynamics
def dynamics(t, state):
    state = state.reshape(2, -1)
    x = state[0]
    x_dot = state[1]
    r.set_root_state(x, x_dot)
    r.pushforward()
    r.pullback()
    x_ddot = r.resolve()
    state_dot = np.concatenate((x_dot, x_ddot), axis=None)
    return state_dot
# --------------------------------------------

# ---------------------------------------------
# solve the diff eq
sol = solve_ivp(dynamics, [0, 60], state_0)
# ---------------------------------------------

# --------------------------------------------
# plot trajectories


for i in range(N):
    plt.plot(sol.y[2 * i], sol.y[2 * i + 1], 'y--')
    plt.plot(x_g[i, 0], x_g[i, 1], 'go')
    plt.plot(x_0[i, 0], x_0[i, 1], 'ro')

plt.axis(np.array([-12, 12, -12, 12]))
plt.gca().set_aspect('equal', 'box')

fig = plt.gcf()
ax = plt.gca()

agents, = plt.plot(sol.y[0: 2 * N: 2, 0], sol.y[1: 2 * N + 1: 2, 0], 'ko')


def init():  # only required for blitting to give a clean slate.
    return agents,


def animate(i):
    nsteps = sol.y.shape[-1]
    agents.set_xdata(sol.y[0: 2 * N: 2, i % nsteps])
    agents.set_ydata(sol.y[1: 2 * N + 1: 2, i % nsteps])
    return agents,


ani = animation.FuncAnimation(
    fig, animate, init_func=init, interval=20, blit=True)

plt.show()
# --------------------------------------------
