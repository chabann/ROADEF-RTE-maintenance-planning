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
# PENALTY_KOEF = 500


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


def check_all_constraints(Instance: dict, PENALTY_KOEF):
    penalty1 = check_schedule(Instance)
    penalty2 = check_resources(Instance)
    penalty3 = check_exclusions(Instance)
    return (penalty1 + penalty2 + penalty3)*PENALTY_KOEF, (penalty1, penalty2, penalty3)  # *PENALTY_KOEF


def check_schedule(Instance: dict):
    shedule_penalty = 0
    Interventions = Instance[INTERVENTIONS_STR]
    for intervention_name, intervention in Interventions.items():
        start_time = intervention[START_STR]
        horizon_end = Instance[T_STR]
        if not (1 <= start_time <= horizon_end):
            if start_time < 1:
                shedule_penalty += 1 - start_time
            else:
                shedule_penalty += start_time - horizon_end
            del intervention[START_STR]
            continue
        else:
            time_limit = int(intervention[TMAX_STR])
            if time_limit < start_time:
                shedule_penalty += start_time - time_limit
                del intervention[START_STR]
                continue
    return shedule_penalty  # * PENALTY_KOEF


def check_resources(Instance: dict):
    resources_penalty = 0
    T_max = Instance[T_STR]
    Resources = Instance[RESOURCES_STR]
    tolerance = 1e-5
    resource_usage = compute_resources(Instance)  # dict on resources and time
    for resource_name, resource in Resources.items():
        for time in range(T_max):
            upper_bound = resource[MAX_STR][time]
            lower_bound = resource[MIN_STR][time]
            worload = resource_usage[resource_name][time]
            if worload > upper_bound + tolerance:
                resources_penalty += worload - upper_bound + tolerance
            if worload < lower_bound - tolerance:
                resources_penalty += lower_bound - tolerance - worload
    return resources_penalty  # * PENALTY_KOEF


def check_exclusions(Instance: dict):
    exclusion_penalty = 0
    result = 0
    Interventions = Instance[INTERVENTIONS_STR]
    Exclusions = Instance[EXCLUSIONS_STR]

    for exclusion in Exclusions.values():

        [intervention_1_name, intervention_2_name, season] = exclusion

        intervention_1 = Interventions[intervention_1_name]
        intervention_2 = Interventions[intervention_2_name]

        if (START_STR not in intervention_1) or (START_STR not in intervention_2):
            continue

        intervention_1_start_time = intervention_1[START_STR]
        intervention_2_start_time = intervention_2[START_STR]

        intervention_1_delta = int(intervention_1[DELTA_STR][intervention_1_start_time - 1])
        intervention_2_delta = int(intervention_2[DELTA_STR][intervention_2_start_time - 1])

        for time_str in Instance[SEASONS_STR][season]:
            time = int(time_str)
            if (intervention_1_start_time <= time < intervention_1_start_time + intervention_1_delta) and \
                    (intervention_2_start_time <= time < intervention_2_start_time + intervention_2_delta):
                exclusion_penalty += 1
        """if exclusion_penalty <= 8:
            result = (np.exp(exclusion_penalty) - 1)*10
        else:
            result = 43 * exclusion_penalty"""
        result = exclusion_penalty * 2  # * PENALTY_KOEF
    return result


def check_and_display(instance, solution, names, PENALTY_KOEF):
    read_solution_from_txt(instance, solution, names)
    penalty, penal_tuple = check_all_constraints(instance, PENALTY_KOEF)
    mean_risk, quantile = compute_objective(instance)
    alpha = instance[ALPHA_STR]

    obj_1 = np.mean(mean_risk)
    tmp = np.zeros(len(quantile))
    obj_2 = np.mean(np.max(np.vstack((quantile - mean_risk, tmp)), axis=0))
    """Interventions = instance[INTERVENTIONS_STR]
    Scenarios = instance[SCENARIO_NUMBER]
    Scenarios = sum(Scenarios) / len(Scenarios)
    PENALTY_KOEFF = Interventions * Scenarios

    penalty *= PENALTY_KOEFF"""
    if penalty > (alpha * obj_1 + (1 - alpha) * obj_2)*0.75:
        penalty = (alpha * obj_1 + (1 - alpha) * obj_2)*0.75
    obj_tot = alpha * obj_1 + (1 - alpha) * obj_2 + penalty
    return obj_tot, penalty, penal_tuple


def main_checker(instance, solution, PENALTY_KOEF):
    # PENALTY_KOEF = (1 + 80 / len(instance[INTERVENTIONS_STR])*300)
    intervention_names = list(instance[INTERVENTIONS_STR].keys())
    total, penalty, pen_tup = check_and_display(instance, solution, intervention_names, PENALTY_KOEF)
    return total, penalty, pen_tup
