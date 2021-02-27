import os
import numpy as np

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


def read_solution_from_txt(Instance: dict, solution, names):
    Interventions = Instance[INTERVENTIONS_STR]
    for iter, start_time_str in enumerate(solution):
        intervention_name = names[iter]
        start_time = int(start_time_str)
        Interventions[intervention_name][START_STR] = start_time


def compute_resources(Instance: dict):
    Interventions = Instance[INTERVENTIONS_STR]
    T_max = Instance[T_STR]
    Resources = Instance[RESOURCES_STR]
    resources_usage = {}
    for resource_name in Resources.keys():
        resources_usage[resource_name] = np.zeros(T_max)
    for intervention_name, intervention in Interventions.items():
        if START_STR not in intervention:
            continue
        start_time = intervention[START_STR]
        start_time_idx = start_time - 1  # - 2  #index of list starts at 0
        intervention_worload = intervention[RESOURCE_CHARGE_STR]
        intervention_delta = int(intervention[DELTA_STR][start_time_idx])

        for resource_name, intervention_resource_worload in intervention_worload.items():
            for time in range(start_time_idx, start_time_idx + intervention_delta):
                if str(time+1) in intervention_resource_worload and str(start_time) in intervention_resource_worload[str(time+1)]:
                    resources_usage[resource_name][time] += intervention_resource_worload[str(time+1)][str(start_time)]

    return resources_usage


def compute_risk_distribution(Interventions: dict, T_max: int, scenario_numbers):
    risk = [scenario_numbers[t] * [0] for t in range(T_max)]
    for intervention in Interventions.values():
        intervention_risk = intervention[RISK_STR]
        if START_STR not in intervention:
            continue
        start_time = intervention[START_STR]
        start_time_idx = int(start_time) - 1  # index for list getter
        delta = int(intervention[DELTA_STR][start_time_idx])
        for time in range(start_time_idx, start_time_idx + delta):
            for i, additional_risk in enumerate(intervention_risk[str(time + 1)][str(start_time)]):
                risk[time][i] += additional_risk
    return risk


def compute_mean_risk(risk, T_max: int, scenario_numbers):
    mean_risk = np.zeros(T_max)
    for t in range(T_max):
        mean_risk[t] = sum(risk[t]) / scenario_numbers[t]
    return mean_risk


def compute_quantile(risk, T_max: int, scenario_numbers, quantile):
    q = np.zeros(T_max)
    for t in range(T_max):
        risk[t].sort()
        q[t] = risk[t][int(np.ceil(scenario_numbers[t] * quantile))-1]
    return q


def compute_objective(Instance: dict):
    T_max = Instance[T_STR]
    scenario_numbers = Instance[SCENARIO_NUMBER]
    Interventions = Instance[INTERVENTIONS_STR]
    quantile = Instance[QUANTILE_STR]
    risk = compute_risk_distribution(Interventions, T_max, scenario_numbers)
    mean_risk = compute_mean_risk(risk, T_max, scenario_numbers)
    q = compute_quantile(risk, T_max, scenario_numbers, quantile)
    return mean_risk, q


def check_and_display(instance, solution, names):
    read_solution_from_txt(instance, solution, names)
    mean_risk, quantile = compute_objective(instance)
    alpha = instance[ALPHA_STR]

    obj_1 = np.mean(mean_risk)
    tmp = np.zeros(len(quantile))
    obj_2 = np.mean(np.max(np.vstack((quantile - mean_risk, tmp)), axis=0))

    obj_tot = alpha * obj_1 + (1 - alpha) * obj_2
    return obj_tot


def check_value(instance, solution):
    intervention_names = list(instance[INTERVENTIONS_STR].keys())
    total = check_and_display(instance, solution, intervention_names)
    return total
