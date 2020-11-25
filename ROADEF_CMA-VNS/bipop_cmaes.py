import math
import numpy as np
from _cma import CMA
from function_checker import main_checker
import random

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


def main_bipop(instance, initial):
    Interventions = instance[INTERVENTIONS_STR]
    dim = len(Interventions)
    Deltas = []
    intervention_names = list(Interventions.keys())
    for i in range(dim):
        Deltas.append(max(Interventions[intervention_names[i]][DELTA_STR]))

    seed = 0
    rng = np.random.RandomState(0)

    bounds = np.array([[1, int(Interventions[intervention_names[i]][TMAX_STR])] for i in range(dim)])
    mean = np.array(initial, dtype='float64')
    sigma = dim * 2 / 5  # 1/5 of the domain width
    optimizer = CMA(mean=mean, sigma=sigma, bounds=bounds, seed=0)

    n_restarts = 0  # A small restart doesn't count in the n_restarts
    small_n_eval, large_n_eval = 0, 0
    popsize0 = optimizer.population_size
    inc_popsize = 2
    best_value = np.inf
    best_solution = []
    best_penalty = 0
    current_value = []
    current_penalty = []
    iteration = 0

    if dim > 100:
        max_n_restart = 1
    else:
        max_n_restart = 2

    poptype = "small"

    while n_restarts < max_n_restart:   # <= 5
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            x = np.around(list(x))
            value, penalty, penalty_tuple = main_checker(instance, list(x))
            solutions.append((x, value))
            iteration += 1

            if value < best_value:
                best_value = value
                best_solution = x
                best_penalty = penalty

            if not (iteration % 500):
                current_value.append(best_value)
                current_penalty.append(best_penalty)

            if not (iteration % 5000):
                print('iteration:', iteration, 'value:', best_value, 'penalty:', best_penalty)

        optimizer.tell(solutions)
        if iteration >= 80000:
            n_restarts = max_n_restart
        if optimizer.should_stop():
            seed += 1
            n_eval = optimizer.population_size * optimizer.generation
            if poptype == "small":
                small_n_eval += n_eval
            else:  # poptype == "large"
                large_n_eval += n_eval

            if small_n_eval < large_n_eval:
                poptype = "small"
                popsize_multiplier = inc_popsize ** n_restarts
                popsize = math.floor(
                    popsize0 * popsize_multiplier ** (rng.uniform() ** 2)
                )
            else:
                poptype = "large"
                n_restarts += 1
                popsize = popsize0 * (inc_popsize ** n_restarts)

            mean = np.array([random.randint(1, int(Interventions[intervention_names[i]][TMAX_STR]))
                             for i in range(dim)], dtype='float64')
            optimizer = CMA(
                mean=mean,
                sigma=sigma,
                bounds=bounds,
                seed=seed,
                population_size=popsize,
            )
            print("Restart CMA-ES with popsize={} ({})".format(popsize, poptype))

    print('Done, Best_value:', best_value, 'Best_penalty:', best_penalty)
    # print('Best solution:', best_solution)
    time = [i for i in range(1, len(current_value) + 1)]
    return current_value, current_penalty, time, best_value, best_penalty, best_solution
