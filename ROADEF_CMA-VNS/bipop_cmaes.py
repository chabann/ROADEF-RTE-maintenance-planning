import math
import numpy as np
from _cma import CMA
from function_checker import main_checker
import random
import time

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


def main_bipop(instance, ini_val, ini_pen, initial, time_limit, penalty_koef):
    start_time = time.time()
    Interventions = instance[INTERVENTIONS_STR]
    dim = len(Interventions)
    Deltas = []
    intervention_names = list(Interventions.keys())
    for i in range(dim):
        Deltas.append(max(Interventions[intervention_names[i]][DELTA_STR]))

    seed = 0
    rng = np.random.RandomState(0)

    bounds = np.array([[1, int(Interventions[intervention_names[i]][TMAX_STR])] for i in range(dim)])
    lower_bounds, upper_bounds = bounds[:, 0], bounds[:, 1]
    # mean = np.array(initial, dtype='float64')
    mean = (lower_bounds + rng.rand(1, dim) * (upper_bounds - lower_bounds))[0]

    # sigma = dim * 2 / 5  # 1/5 of the domain width
    sigma = (max(upper_bounds) - min(lower_bounds))/2 * 2/5
    optimizer = CMA(mean=mean, sigma=sigma, bounds=bounds, seed=0)

    n_restarts = 0  # A small restart doesn't count in the n_restarts
    small_n_eval, large_n_eval = 0, 0
    popsize0 = optimizer.population_size
    inc_popsize = 3
    # best_value = np.inf
    best_value = ini_val
    best_solution = initial
    # best_penalty = 0
    best_penalty = ini_pen
    current_value = []
    current_penalty = []
    iteration = 0
    solutions = []

    max_n_restart = 25

    #poptype = "small"
    poptype = "large"

    while n_restarts < max_n_restart:
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            x = np.around(list(x))
            value, penalty, penalty_tuple = main_checker(instance, list(x), penalty_koef)
            solutions.append((x, value))
            iteration += 1

            if value < best_value:
                best_value = value
                best_solution = x
                best_penalty = penalty

            if not (iteration % 150):
                current_value.append(best_value)
                current_penalty.append(best_penalty)

        optimizer.tell(solutions)
        if time.time() - start_time >= time_limit:
            n_restarts = max_n_restart
            break

        if optimizer.should_stop():
            seed += 1
            n_eval = optimizer.population_size * optimizer.generation * 2
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

            mean = (lower_bounds + rng.rand(1, dim) * (upper_bounds - lower_bounds))[0]

            optimizer = CMA(
                mean=mean,
                sigma=sigma,
                bounds=bounds,
                seed=seed,
                population_size=popsize,
            )

    print('Done CMA-ES, Best_value:', best_value, 'Best_penalty:', best_penalty)

    ar_time = [i for i in range(1, len(current_value) + 1)]
    print(time.time() - start_time, 'seconds for CMA')
    return current_value, current_penalty, ar_time, best_value, best_penalty, best_solution, solutions
