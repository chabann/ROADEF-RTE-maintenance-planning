from function_checker import main_checker

import random
import os
import matplotlib.pyplot as plt

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
PENALTY_KOEF = 500


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


def local_search(best, max_no_improv, neighborhood_size, instance):
    count = 0

    while count < max_no_improv:
        candidate = {}
        candidate["vector"] = [v for v in best["vector"]]

        for _ in range(neighborhood_size):
            stochastic_two_opt(candidate["vector"])

        candidate["cost"], candidate["penalty"], candidate["penalty_tuple"] = main_checker(instance, candidate["vector"])

        if (candidate["cost"] < best["cost"]) & (candidate["penalty"] <= best["penalty"]):
            count, best = 0, candidate
        else:
            count += 1

    return best


def search(neighborhoods, max_no_improv, max_no_improv_ls, instance, intervention_names, Interventions, dim, ini_vector):
    best = {}
    # best["vector"] = [random.randint(1, int(Interventions[intervention_names[i]][TMAX_STR])) for i in range(dim)]
    best["vector"] = ini_vector
    best["cost"], best["penalty"], best["penalty_tuple"] = main_checker(instance, best["vector"])
    iter_, count = 0, 0
    info = {}
    info["function"] = []
    info["penalty"] = []
    info["time"] = []

    while count < max_no_improv:
        for neigh in neighborhoods:
            candidate = {}
            candidate["vector"] = [v for v in best["vector"]]

            for _ in range(neigh):
                stochastic_two_opt(candidate["vector"])

            candidate["cost"], candidate["penalty"], candidate["penalty_tuple"] = main_checker(instance, candidate["vector"])
            candidate = local_search(candidate, max_no_improv_ls, neigh, instance)
            if not (iter_ % 50):
                print("iter:", iter_ + 1, "neigh:", neigh, "best:", best["cost"], "penalty:", best["penalty_tuple"])
            iter_ += 1

            if (candidate["cost"] < best["cost"]) & (candidate["penalty"] <= best["penalty"]):
                best, count = candidate, 0
                print("New best, restarting neighborhood search")
                break
            else:
                count += 1
            info["function"].append(best["cost"])
            info["penalty"].append(best["penalty"])
            info["time"].append(iter_)
    return best, info


def main_vns(instance, initial_vol):
    Interventions = instance[INTERVENTIONS_STR]
    dim = len(Interventions)
    Deltas = []
    intervention_names = list(Interventions.keys())

    for i in range(dim):
        Deltas.append(max(Interventions[intervention_names[i]]['Delta']))

    max_no_improv = 120  # 50  # 70
    max_no_improv_ls = 150  # 70  # 90
    neighborhoods = list(range(30))
    best, plot_info = search(neighborhoods, max_no_improv, max_no_improv_ls, instance,
                             intervention_names, Interventions, dim, initial_vol)
    # print("Done. Best Solution:", best["cost"], best["vector"], best["penalty_tuple"])
    print("Done. Best Solution:", best["cost"], best["penalty_tuple"])

    return plot_info["time"], plot_info["function"], best["cost"], best["vector"], best["penalty_tuple"]
