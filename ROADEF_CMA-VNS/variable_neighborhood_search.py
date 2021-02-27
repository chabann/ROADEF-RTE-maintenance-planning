from function_checker import main_checker
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


def stochastic_two_opt(perm):
    randlimit = len(perm) - 1
    c1, c2 = random.randint(0, randlimit), random.randint(0, randlimit)
    exclude = [c1]
    exclude.append(randlimit if c1 == 0 else c1 - 1)
    exclude.append(0 if c1 == randlimit else c1 + 1)

    while c2 in exclude:
        c2 = random.randint(0, randlimit)

    c1, c2 = c2, c1 if c2 < c1 else None
    perm[c1:c2] = perm[c1:c2][::-1]
    return perm


def local_search(best, max_no_improv, neighborhood_size, instance, penalty_koef):
    count = 0
    while count < max_no_improv:
        candidate = {}
        candidate["vector"] = [v for v in best["vector"]]

        for _ in range(neighborhood_size):
            stochastic_two_opt(candidate["vector"])

        candidate["cost"], candidate["penalty"], candidate["penalty_tuple"] = \
            main_checker(instance, candidate["vector"], penalty_koef)

        if (candidate["cost"] < best["cost"]) & (candidate["penalty"] <= best["penalty"]):
            count, best = 0, candidate
        else:
            count += 1
    return best


def search(neighborhoods, max_no_improv, max_no_improv_ls, instance, ini_vector, time_limit, start_time, penalty_koef):
    best = {}
    best["vector"] = ini_vector
    best["cost"], best["penalty"], best["penalty_tuple"] = main_checker(instance, best["vector"], penalty_koef)
    iter_, count = 0, 0
    info = {}
    info["function"] = []
    info["penalty"] = []
    info["time"] = []

    while (time.time() - start_time) < time_limit:
        for neigh in neighborhoods:

            if (time.time() - start_time) >= time_limit:
                break

            candidate = {}
            candidate["vector"] = [v for v in best["vector"]]

            for _ in range(neigh):
                stochastic_two_opt(candidate["vector"])

            candidate["cost"], candidate["penalty"], candidate["penalty_tuple"] = \
                main_checker(instance, candidate["vector"], penalty_koef)
            candidate = local_search(candidate, max_no_improv_ls, neigh, instance, penalty_koef)
            iter_ += 1

            if (candidate["cost"] < best["cost"]) & (candidate["penalty"] <= best["penalty"]):
                best, count = candidate, 0
                break
            else:
                count += 1
            info["function"].append(best["cost"])
            info["penalty"].append(best["penalty"])
            info["time"].append(iter_)

    return best, info


def main_vns(instance, initial_vol, time_limit, penalty_koef):
    start_time = time.time()
    Interventions = instance[INTERVENTIONS_STR]
    dim = len(Interventions)
    Deltas = []
    intervention_names = list(Interventions.keys())

    for i in range(dim):
        Deltas.append(max(Interventions[intervention_names[i]]['Delta']))
    numb_neigh = dim

    max_no_improv = 50
    max_no_improv_ls = 70
    neighborhoods = list(range(numb_neigh))
    best, plot_info = search(neighborhoods, max_no_improv, max_no_improv_ls, instance,
                             initial_vol, time_limit, start_time, penalty_koef)

    print("Done VNS. Best Solution:", best["cost"], "penalty:", best["penalty"])
    print(time.time() - start_time, 'seconds for VNS')

    return plot_info["time"], plot_info["function"], best["cost"], best["vector"], best["penalty"]
