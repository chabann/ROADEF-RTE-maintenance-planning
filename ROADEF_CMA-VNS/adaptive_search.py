import random
from function_checker import main_checker
import os
import time

# Global variables
CUR_DIR = os.getcwd()
PAR_DIR = os.path.dirname(CUR_DIR)
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


def rand_in_bounds(minimum, maximum):
    return minimum + ((maximum - minimum) * random.random())


def random_vector(minmax):
    """
    _Essentially_ similar to the one in [2.2]
    """
    i = 0
    limit = len(minmax)
    random_vector = [0 for i in range(limit)]

    for i in range(limit):
        random_vector[i] = rand_in_bounds(minmax[i][0], minmax[i][1])

    return random_vector


def take_step(minmax, current, step_size):
    limit = len(current)
    position = [0 for i in range(limit)]

    for i in range(limit):
        minimum = max(minmax[i][0], current[i] - step_size)
        maximum = min(minmax[i][1], current[i] + step_size)
        position[i] = round(rand_in_bounds(minimum, maximum))

    return position


def large_step_size(iter_count, step_size, s_factor, l_factor, iter_mult):
    if iter_count > 0 and iter_count % iter_mult == 0:
        return step_size * l_factor
    else:
        return step_size * s_factor


def take_steps(bounds, current, step_size, big_stepsize, instance):
    step, big_step = {}, {}
    step["vector"] = take_step(bounds, current["vector"], step_size)
    step["cost"], step["penalty"], step["penalty_tuple"] = main_checker(instance, step["vector"], )
    big_step["vector"] = take_step(bounds, current["vector"], big_stepsize)
    big_step["cost"], big_step["penalty"], big_step["penalty_tuple"] = main_checker(instance, big_step["vector"])
    return step, big_step


def search(max_iter, bounds, init_factor, s_factor, l_factor, iter_mult, max_no_impr, intervention_names,
           dim, instance, initial):
    Interventions = instance[INTERVENTIONS_STR]
    step_size = (bounds[0][1] - bounds[0][0]) * init_factor
    current, count = {}, 0
    current["vector"] = initial
    current["cost"], current["penalty"], current["penalty_tuple"] = main_checker(instance, current["vector"])
    time = [1]
    values = [current["cost"]]
    for i in range(max_iter):
        big_stepsize = large_step_size(i, step_size, s_factor, l_factor, iter_mult)
        step, big_step = take_steps(bounds, current, step_size, big_stepsize, instance)

        if not (i % 20):
            time.append(time[-1]+1)
            values.append(current["cost"])

        if (step["cost"] <= current["cost"] or big_step["cost"] <= current["cost"]) & \
                (step["penalty"] <= current["penalty"] or big_step["penalty"] <= current["penalty"]):
            if big_step["cost"] <= step["cost"]:
                step_size, current = big_stepsize, big_step
            else:
                current = step
            count = 0
        else:
            count += 1
            if count >= max_no_impr:
                count, stepSize = 0, (step_size / s_factor)
    return current, time, values


def main_as(instance, initial, time_limit, penalty_koef):
    Interventions = instance[INTERVENTIONS_STR]

    dim = len(Interventions)
    Time = instance[T_STR]
    Deltas = []
    intervention_names = list(Interventions.keys())
    for i in range(dim):
        Deltas.append(max(Interventions[intervention_names[i]]['Delta']))

    bounds = [[1, int(Interventions[intervention_names[i]][TMAX_STR])] for i in range(dim)]
    max_iter = 4000
    init_factor = 0.1
    s_factor = Time/16
    l_factor = Time/8
    iter_mult = 10
    max_no_impr = 30

    best, time, values = search(max_iter, bounds, init_factor, s_factor, l_factor, iter_mult,
                                max_no_impr, intervention_names, dim, instance, initial, penalty_koef, time_limit)
    print("Done AS. Best Solution:", best["cost"], "penalty:", best["penalty"])
    return best["cost"], best["vector"], best["penalty"], time, values
