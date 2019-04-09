# Robotarium example: formation preservation tasks with two subteams
# @author Anqi Li
# @date April 8, 2019

import rps.robotarium as robotarium
import rps.utilities.graph as graph
import rps.utilities.transformations as transformations
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import at_pose
from rps.utilities.controllers import *

import time

import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import pdist, squareform

from rmp import RMPRoot, RMPNode
from rmp_leaf import CollisionAvoidanceCentralized, GoalAttractorUni, FormationCentralized, Damper

from matplotlib import pyplot as plt

# Instantiate Robotarium object

# safety distance
R_safety = 0.2

# number of agents
N_1 = 5 # number of agents for subteam 1
N_2 = 4 # number of agents for subteam 2
N = N_1 + N_2

# initial configuration
theta_1 = np.arange(0, 2 * np.pi, 2 * np.pi / N_1)
x_0_1 = np.stack((- np.cos(theta_1) * 0.4 - 0.5, - np.sin(theta_1) * 0.3 - 0.5, theta_1))
theta_2 = np.arange(0, 2 * np.pi, 2 * np.pi / N_2)
x_0_2 = np.stack((np.cos(theta_2) * 0.3 + 0.5, np.sin(theta_2) * 0.4 + 0.5, theta_2))
x_0 = np.concatenate((x_0_1, x_0_2), axis=1)

# laplacian for formation graph
L = np.zeros((N, N))

L[0, 1:N_1] = 1
for i in range(N_1 - 2):
    L[i+1, i+2] = 1

L[N_1, N_1 + 1: ] = 1
for i in range(N_1, N - 2):
    L[i+1, i+2] = 1


# distances for formation graph
thetas1 = np.arange(0, 2 * np.pi, 2 * np.pi / N_1)
dd1 = 0.3
coords1 = np.stack((np.cos(thetas1) * dd1, np.sin(thetas1) * dd1), axis=1)
dists1 = squareform(pdist(coords1))

thetas2 = np.arange(0, 2 * np.pi, 2 * np.pi / N_2)
dd2 = 0.3
coords2 = np.stack((np.cos(thetas2) * dd2, np.sin(thetas2) * dd2), axis=1)
dists2 = squareform(pdist(coords2))

dists = np.zeros((N, N))
dists[: N_1, : N_1] = dists1
dists[N_1 :, N_1: ] = dists2


# goals for the two subteams
g_1 = np.array([-0.6, -0.6])
g_2 = np.array([0.6, 0.6])


# random leader robots
leader1 = np.random.randint(N_1)
leader2 = np.random.randint(N_1, N)



# intialize the robotarium
rb = robotarium.Robotarium(number_of_agents=N, show_figure=True, save_data=False, update_time=0.1)

# the algorithm uses single-integrator dynamics, so we'll need these mappings.
si_to_uni_dyn, uni_to_si_states = transformations.create_single_integrator_to_unicycle()

# barrier certificate to avoid collisions. used for driving to the initial configs, 
# not used during the algorthim
si_barrier_cert = create_single_integrator_barrier_certificate(N)



# --------------------------------------------------------------------------------------
# build the RMPtree

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

def create_mappings(i):
    phi = lambda y, i=i: np.array([[y[2 * i, 0]], [y[2 * i + 1, 0]]])
    J = lambda y, i=i: np.concatenate((
            np.zeros((2, 2 * i)),
            np.eye(2),
            np.zeros((2, 2 * (N - i - 1)))), axis=1)
    J_dot = lambda y, y_dot: np.zeros((2, 2 * N))

    return phi, J, J_dot



r = RMPRoot('root')

robots = []


for i in range(N):
    phi, J, J_dot = create_mappings(i)
    robot = RMPNode('robot_' + str(i), r, phi, J, J_dot)
    robots.append(robot)


dps = []
for i in range(N):
    dp = Damper(
        'dp_robot_' + str(i),
        robots[i],
        w=0.01)
    dps.append(dp)


pairs = []
iacas = []
fcs = []
weight_fc = 10
for i in range(N):
    for j in range(N):
        if i >= j:
            continue
        phi, J, J_dot = create_mappings_pair(i, j)
        pair = RMPNode('pair_' + str(i) + '_' + str(j), r, phi, J, J_dot)
        pairs.append(pair)

        iaca = CollisionAvoidanceCentralized(
            'cac_' + pair.name,
            pair,
            R=R_safety,
            eta=0.5)
        iacas.append(iaca)

        if L[i, j]:
            fc = FormationCentralized(
                'fc_robot_' + str(i) + '_robot_' + str(j),
                pair,
                d=dists[i, j],
                w=weight_fc,
                eta=5,
                gain=1)
            fcs.append(fc)


ga1 = GoalAttractorUni(
    'ga_leader_0',
    robots[leader1],
    g_1,
    w_u = 10,
    w_l = 0.01,
    sigma = 0.1,
    alpha = 1,
    gain = 1,
    eta = 1)

ga2 = GoalAttractorUni(
    'ga_leader_1',
    robots[leader2],
    g_2,
    w_u = 10,
    w_l = 0.01,
    sigma = 0.1,
    alpha = 1,
    gain = 1,
    eta = 1)

# ------------------------------------------------------
# drive to the initial configurations

x_uni = rb.get_poses()
si_velocities = np.zeros((2, N))
rb.set_velocities(np.arange(N), si_to_uni_dyn(si_velocities, x_uni))
rb.step()


for k in range(3000):
    x_uni = rb.get_poses()
    x_si = x_uni[:2, :]

    if np.size(at_pose(x_uni, x_0, rotation_error=100)) == N:
        print('done!')
        break

    si_velocities = (x_0[:2, :] - x_si)
    si_velocities = si_barrier_cert(si_velocities, x_si)
    rb.set_velocities(np.arange(N), si_to_uni_dyn(si_velocities, x_uni))
    try:
        rb.step()
    except:
        rb.call_at_scripts_end()
        exit(0)


# --------------------------------------------------------------

# graphics

gh1, = plt.plot([g_1[0]], [g_1[1]], 'r*', markersize=50)
gh2, = plt.plot([g_2[0]], [g_2[1]], '*', color=[0, 0.45, 0.74], markersize=50)

plt.text(g_1[0] + 0.1, g_1[1], 'A', fontsize=25)
plt.text(g_2[0] + 0.1, g_2[1], 'B', fontsize=25)

ehs = []
count = 0
for i in range(N):
    for j in range(N):
        if L[i, j]:
            if i < N_1:
                eh, = plt.plot([x_uni[0, i], x_uni[0, j]], [x_uni[1, i], x_uni[1, j]], 'r-', linewidth=3)
            else:
                eh, = plt.plot([x_uni[0, i], x_uni[0, j]], [x_uni[1, i], x_uni[1, j]], '-', color=[0, 0.45, 0.74], linewidth=3)
            ehs.append(eh)
            count += 1


dt = rb.time_step
si_velocities = np.zeros((2, N))


# ----------------------------------------------------

time.sleep(1)


dt = rb.time_step
si_velocities = np.zeros((2, N))
magnitude_limit = 0.4


def at_position(state, pose, position_error=0.02):
    """Checks whether robots are "close enough" to goals

    states: 3xN numpy array (of unicycle states)
    poses: 3xN numpy array (of desired states)

    -> 1xN numpy index array (of agents that are close enough)
    """
    # Calculate position errors
    pes = norm(state - pose)

    # Determine which agents are done
    done = pes <= position_error

    return done


# -----------------------------------------------
# main simulation

for k in range(6000):

    # Get the poses of the robots and convert to single-integrator poses
    x_uni = rb.get_poses()
    x_si = uni_to_si_states(x_uni)
    x = x_si.T.flatten()
    x_dot = si_velocities.T.flatten()

    r.set_root_state(x, x_dot)
    r.pushforward()
    r.pullback()
    try:
        a = r.resolve()
        si_accelerations = a.reshape(-1, 2).T
        # simulate double-integrator dynamics
        si_velocities = si_velocities + si_accelerations * dt
        norms = norm(si_velocities, axis=0)
        idxs_to_normalize = (norms > magnitude_limit)
        si_velocities[:, idxs_to_normalize] *= magnitude_limit * (1 / norms[idxs_to_normalize])
    except:
        si_velocities = np.zeros((2, N))
        print(x_si)
        print('Warning: no sol found, emergency break')
        break

    # Set the velocities of agents 1,...,N
    rb.set_velocities(np.arange(N), si_to_uni_dyn(si_velocities, x_uni))


    # if both teams at goal, swap goals
    if at_position(x_si[:, leader1], g_1) and at_position(x_si[:, leader2], g_2):
        print('change goal')
        g_1 = -g_1
        g_2 = -g_2
        ga1.update_goal(g_1)
        ga2.update_goal(g_2)
        gh1.set_data([g_1[0]], [g_1[1]])
        gh2.set_data([g_2[0]], [g_2[1]])

    count = 0
    for i in range(N):
        for j in range(N):
            if L[i, j]:
                ehs[count].set_data([x_uni[0, i], x_uni[0, j]], [x_uni[1, i], x_uni[1, j]])
                count += 1


    # Iterate the simulation
    try:
        rb.step()
    except:
        rb.call_at_scripts_end()
        exit(0)


# input('press enter')
# Always call this function at the end of your script!!!!
rb.call_at_scripts_end()
