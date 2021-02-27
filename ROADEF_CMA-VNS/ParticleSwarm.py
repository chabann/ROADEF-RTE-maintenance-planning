import numpy as np
import copy
import numpy.random as rnd
import ParticleSwarmUtility as PSU
from function_checker import main_checker
import time
import random

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


def particleswarm(f, bounds, p, c1, c2, vmax, instance, intervention_names, initial,
                  time_limit, start_time, penalty_koef, population):
    Interventions = instance[INTERVENTIONS_STR]
    d, particle_pos, particle_best, swarm_best, particle_velocity, local_best, pos_val, pos_pen = \
        PSU.initiation(f, bounds, p, instance, intervention_names, initial, penalty_koef, population)

    values = []
    c3 = c1+c2
    K = 2/(abs(2-c3-np.sqrt(abs((c3**2)-(4*c3)))))
    f_swarm_best, f_swarm_pen, f_swarm_pen_tup = f(instance, swarm_best, penalty_koef)

    iter = -1
    while (time.time() - start_time) < time_limit:
        iter += 1
        if (time.time() - start_time) >= time_limit:
            break

        if not (iter == 1000):
            for j in range(p):
                particle_velocity[j] = [(rnd.uniform(-abs(bounds[i][1]-bounds[i][0]),
                                                     abs(bounds[i][1]-bounds[i][0]))) for i in range(d)]

        if not (iter == 500):
            for j in range(p):
                particle_pos[j] = [random.randint(1, int(Interventions[intervention_names[i]][TMAX_STR])) for i in
                                   range(d)]
                if j == (p - 1):
                    particle_pos[j] = initial
                particle_velocity[j] = [rnd.uniform(-abs(bounds[i][1] - bounds[i][0]),
                                                    abs(bounds[i][1] - bounds[i][0])) for i in range(d)]

        for i in range(p):
            rp, rg = rnd.uniform(0, 1, 2)
            particle_velocity[i, :] += (c1*rp*(particle_best[i, :]-particle_pos[i, :]))
            particle_velocity[i, :] += (c2*rg*(local_best[i, :]-particle_pos[i, :]))
            particle_velocity[i, :] = particle_velocity[i, :]*K

            if particle_velocity[i].any() > vmax:
                particle_velocity[i, :] = vmax
            particle_pos[i][:] += particle_velocity[i, :]
            particle_pos[i] = PSU.withinbounds(bounds, particle_pos[i])
            particle_fitness, particle_penalty, part_tuple = f(instance, particle_pos[i], penalty_koef)
        
            if (particle_fitness < pos_val[i]) & (particle_penalty <= pos_pen[i]):
                particle_best[i, :] = particle_pos[i, :]
                pos_val[i] = particle_fitness

                if (particle_fitness < f_swarm_best) & (particle_penalty <= f_swarm_pen):
                    f_swarm_best = particle_fitness
                    f_swarm_pen = particle_penalty
                    swarm_best = particle_pos[i]

        local_best = PSU.local_best_get(particle_pos, pos_val, p)

        if not (iter % 100):
            values.append(f_swarm_best)

    print('Done PSO. Best Solution:', f_swarm_best, 'penalty:', f_swarm_pen)
    return f_swarm_best, swarm_best, f_swarm_pen, values


def pso_main(instance, initial, time_limit, penalty_koef):
    start_time = time.time()
    Interventions = instance[INTERVENTIONS_STR]

    dim = len(Interventions)
    Deltas = []

    intervention_names = list(Interventions.keys())
    for i in range(dim):
        Deltas.append(max(Interventions[intervention_names[i]]['Delta']))

    f = main_checker
    bounds = [[1, int(Interventions[intervention_names[i]][TMAX_STR])] for i in range(dim)]

    vmax = (max(max(bounds)) - min(min(bounds)))*0.75
    c1 = 2.5
    c2 = 1.5

    p = dim
    """pop_len = len(population)
    if pop_len < p:
        for i in range(pop_len, p):
            population.append([random.randint(1, int(Interventions[intervention_names[i]][TMAX_STR])) for i in range(dim)])
    else:
        if pop_len > p:
            population = population[0:p]
    population[p-1] = initial
    """
    population = []
    for i in range(dim):
        population.append([random.randint(1, int(Interventions[intervention_names[i]][TMAX_STR])) for i in range(dim)])

    best, best_sol, best_pen, values = particleswarm(f, bounds, p, c1, c2, vmax, instance, intervention_names,
                                                     initial, time_limit, start_time, penalty_koef, population)
    ar_time = list(range(len(values)))
    print(time.time() - start_time, 'seconds for PSO')
    return ar_time, values, best, best_sol, best_pen
