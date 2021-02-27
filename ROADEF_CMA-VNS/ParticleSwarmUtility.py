import numpy as np
import copy
import numpy.random as rnd
import random
import time

# Global variables
RESOURCES_STR = 'Resources'
SEASONS_STR = 'Seasons'
INTERVENTIONS_STR = 'Interventions'
EXCLUSIONS_STR = 'Exclusions'
T_STR = 'T'
SCENARIO_NUMBER = 'Scenarios_number'
RESOURCE_CHARGE_STR = 'workload'
TMAX_STR = 'tmax'
DELTA_STR = 'Delta'
MAX_STR = 'max'
MIN_STR = 'min'
RISK_STR = 'risk'
START_STR = 'start'
QUANTILE_STR = "Quantile"
ALPHA_STR = "Alpha"


def local_best_get(particle_pos, particle_pos_val, p):
    local_best = [0 for _ in range(p)]  # creating empty local best list
    for j in range(p):  # finding the best particle in each neighbourhood
        # and storing it in 'local_best'
        local_vals = np.zeros(3)
        local_vals[0] = particle_pos_val[j - 2]
        local_vals[1] = particle_pos_val[j - 1]
        local_vals[2] = particle_pos_val[j]
        min_index = int(np.argmin(local_vals))
        local_best[j - 1] = particle_pos[min_index + j - 2][:]
    return np.array(local_best)


def initiation(f, bounds, p, instance, intervention_names, initial, penalty_koef, particle_pos):
    Interventions = instance[INTERVENTIONS_STR]
    dim = len(bounds)

    # particle_pos = [[0 for j in range(dim)] for i in range(p)]
    particle_velocity = particle_pos[:]
    particle_pos_val = [0 for i in range(p)]  # empty value array
    particle_pos_pen = [0 for i in range(p)]  # empty penalty array

    for j in range(p):  # iterating over the number of particles
        particle_pos[j] = [random.randint(1, int(Interventions[intervention_names[i]][TMAX_STR])) for i in range(dim)]
        if j == (p-1):
            particle_pos[j] = initial

        particle_pos_val[j], particle_pos_pen[j], particle_tuple = f(instance, particle_pos[j], penalty_koef)

        particle_velocity[j] = [rnd.uniform(-abs(bounds[i][1] - bounds[i][0]),
                                            abs(bounds[i][1] - bounds[i][0])) for i in range(dim)]

    local_best = local_best_get(particle_pos, particle_pos_val, p)
    index = np.argmin(particle_pos_val)
    swarm_best = particle_pos[index][:]  # getting the lowest particle value

    particle_best = copy.deepcopy(particle_pos)  # setting all particles current positions to best
    return dim, np.array(particle_pos, dtype='float64'), np.array(particle_best), np.array(swarm_best),\
           np.array(particle_velocity), np.array(local_best), np.array(particle_pos_val), np.array(particle_pos_pen)


def withinbounds(bounds, particle_pos):
    for i in range(len(bounds)):
        if particle_pos[i] < bounds[i][0]:  # if particle is less than lower bound
            particle_pos[i] = bounds[i][0]
        elif particle_pos[i] > bounds[i][1]:  # if particle is more than higher bound
            particle_pos[i] = bounds[i][1]
        particle_pos[i] = round(particle_pos[i])
    return particle_pos
