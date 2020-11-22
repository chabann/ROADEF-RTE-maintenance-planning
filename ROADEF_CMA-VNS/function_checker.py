import os
import numpy as np
from json_reader import read_json

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
    """Read a txt formated Solution file, and store the solution informations in Instance"""
    Interventions = Instance[INTERVENTIONS_STR]
    for iter, start_time_str in enumerate(solution):
        intervention_name = names[iter]
        start_time = int(start_time_str)
        Interventions[intervention_name][START_STR] = start_time

################################
# Results processing ##########
################################

# Compute effective worload
def compute_resources(Instance: dict):
    """Compute effective workload (i.e. resources consumption values) for every resource and every time step"""

    # Retrieve usefull infos
    Interventions = Instance[INTERVENTIONS_STR]
    T_max = Instance[T_STR]
    Resources = Instance[RESOURCES_STR]
    # Init resource usage dictionnary for each resource and time
    resources_usage = {}
    for resource_name in Resources.keys():
        resources_usage[resource_name] = np.zeros(T_max)
    # Compute value for each resource and time step
    for intervention_name, intervention in Interventions.items():
        # start time should be defined (already checked in scheduled constraint checker)
        if not START_STR in intervention:
            continue
        start_time = intervention[START_STR]
        start_time_idx = start_time - 1 # - 2  #index of list starts at 0
        intervention_worload = intervention[RESOURCE_CHARGE_STR]
        intervention_delta = int(intervention[DELTA_STR][start_time_idx])
        # compute effective worload
        for resource_name, intervention_resource_worload in intervention_worload.items():
            for time in range(start_time_idx, start_time_idx + intervention_delta):
                # null values are not available
                if str(time+1) in intervention_resource_worload and str(start_time) in intervention_resource_worload[str(time+1)]:
                    resources_usage[resource_name][time] += intervention_resource_worload[str(time+1)][str(start_time)]

    return resources_usage


# Retrieve effective risk distribution given starting times solution
def compute_risk_distribution(Interventions: dict, T_max: int, scenario_numbers):
    """Compute risk distributions for all time steps, given the interventions starting times"""

    # print('\tComputing risk...')
    # Init risk table
    risk = [scenario_numbers[t] * [0] for t in range(T_max)]
    # Compute for each intervention independently
    for intervention in Interventions.values():
        # Retrieve Intervention's usefull infos
        intervention_risk = intervention[RISK_STR]
        # start time should be defined (already checked in scheduled constraint checker)
        if not START_STR in intervention:
            continue
        start_time = intervention[START_STR]
        start_time_idx = int(start_time) - 1  # index for list getter
        delta = int(intervention[DELTA_STR][start_time_idx])
        for time in range(start_time_idx, start_time_idx + delta):
            for i, additional_risk in enumerate(intervention_risk[str(time + 1)][str(start_time)]):
                risk[time][i] += additional_risk
    # print('\tDone')

    return risk




# Compute mean for each period
def compute_mean_risk(risk, T_max: int, scenario_numbers):
    """Compute mean risk values over each time period"""

    # print('\tComputing mean risk...')
    # Init mean risk
    mean_risk = np.zeros(T_max)
    # compute mean
    for t in range(T_max):
        mean_risk[t] = sum(risk[t]) / scenario_numbers[t]
    # print('\tDone')

    return mean_risk


# Compute quantile for each period
def compute_quantile(risk, T_max: int, scenario_numbers, quantile):
    """Compute Quantile values over each time period"""
    q = np.zeros(T_max)
    for t in range(T_max):
        risk[t].sort()
        q[t] = risk[t][int(np.ceil(scenario_numbers[t] * quantile))-1]
    return q


# Compute both objectives: mean risk and quantile
def compute_objective(Instance: dict):
    """Compute objectives (mean and expected excess)"""
    T_max = Instance[T_STR]
    scenario_numbers = Instance[SCENARIO_NUMBER]
    Interventions = Instance[INTERVENTIONS_STR]
    quantile = Instance[QUANTILE_STR]
    # Retrieve risk final distribution
    risk = compute_risk_distribution(Interventions, T_max, scenario_numbers)
    # Compute mean risk
    mean_risk = compute_mean_risk(risk, T_max, scenario_numbers)
    # Compute quantile
    q = compute_quantile(risk, T_max, scenario_numbers, quantile)
    # print('Done')

    return mean_risk, q


def check_all_constraints(Instance: dict):
    """Run all constraint checks"""

    penalty1 = check_schedule(Instance)
    penalty2 = check_resources(Instance)
    penalty3 = check_exclusions(Instance)

    return penalty1 + penalty2 + penalty3, (penalty1, penalty2, penalty3)  # *PENALTY_KOEF


def check_schedule(Instance: dict):
    """Check schedule constraints"""

    shedule_penalty = 0
    Interventions = Instance[INTERVENTIONS_STR]
    for intervention_name, intervention in Interventions.items():
        # if START_STR not in intervention:
            # shedule_penalty += 1
            # continue
        start_time = intervention[START_STR]
        horizon_end = Instance[T_STR]
        if not (1 <= start_time <= horizon_end):
            if start_time < 1:
                shedule_penalty += 1 - start_time
            else:
                shedule_penalty += start_time - horizon_end
            del intervention[START_STR]
            continue

        time_limit = int(intervention[TMAX_STR])
        if time_limit < start_time:
            shedule_penalty += start_time - time_limit
            # print('ERROR: Schedule constraint 4.1.3: Intervention ' + intervention_name + ' realization exceeds time limit.'
            # + ' It starts at ' + str(start_time) + ' while time limit is ' + str(time_limit) + '.')
            # Remove start time to avoid later access errors
            del intervention[START_STR]
            continue
    return shedule_penalty * PENALTY_KOEF


def check_resources(Instance: dict):
    """Check resources constraints"""
    resources_penalty = 0
    T_max = Instance[T_STR]
    Resources = Instance[RESOURCES_STR]
    # Bounds are checked with a tolerance value
    tolerance = 1e-5
    # Compute resource usage
    resource_usage = compute_resources(Instance)  # dict on resources and time
    # Compare bounds to usage
    for resource_name, resource in Resources.items():
        for time in range(T_max):
            # retrieve bounds values
            upper_bound = resource[MAX_STR][time]
            lower_bound = resource[MIN_STR][time]
            # Consumed value
            worload = resource_usage[resource_name][time]
            # Check max
            if worload > upper_bound + tolerance:
                resources_penalty += worload - upper_bound + tolerance
                # resources_penalty += 1
            if worload < lower_bound - tolerance:
                resources_penalty += lower_bound - tolerance - worload
                # resources_penalty += 1
    return resources_penalty * PENALTY_KOEF


def check_exclusions(Instance: dict):
    """Check exclusions constraints"""
    exclusion_penalty = 0
    result = 0
    Interventions = Instance[INTERVENTIONS_STR]
    Exclusions = Instance[EXCLUSIONS_STR]

    for exclusion in Exclusions.values():

        [intervention_1_name, intervention_2_name, season] = exclusion

        intervention_1 = Interventions[intervention_1_name]
        intervention_2 = Interventions[intervention_2_name]

        if (not START_STR in intervention_1) or (not START_STR in intervention_2):
            continue

        intervention_1_start_time = intervention_1[START_STR]
        intervention_2_start_time = intervention_2[START_STR]

        intervention_1_delta = int(intervention_1[DELTA_STR][intervention_1_start_time - 1])
        intervention_2_delta = int(intervention_2[DELTA_STR][intervention_2_start_time - 1])

        for time_str in Instance[SEASONS_STR][season]:
            time = int(time_str)
            if (intervention_1_start_time <= time < intervention_1_start_time + intervention_1_delta) and (intervention_2_start_time <= time < intervention_2_start_time + intervention_2_delta):
                exclusion_penalty += 1
        if exclusion_penalty >= 15:
            result = np.exp(exclusion_penalty) - 1
        else:
            result = (np.exp(exclusion_penalty) - 1) * PENALTY_KOEF
    return result


def display_basic(Instance: dict, mean_risk, quantile):
    """Print main infos"""
    alpha = Instance[ALPHA_STR]
    q = Instance[QUANTILE_STR]
    print('Instance infos:')
    print('\tInterventions number: ', len(Instance[INTERVENTIONS_STR]))
    print('\tScenario numbers: ', len(Instance[SCENARIO_NUMBER]))
    print('Solution evaluation:')
    obj_1 = np.mean(mean_risk)
    print('\tObjective 1 (mean risk): ', obj_1)
    tmp = np.zeros(len(quantile))
    obj_2 = np.mean(np.max(np.vstack((quantile - mean_risk, tmp)), axis=0))
    print('\tObjective 2 (expected excess  (Q' + str(q) + ')): ', obj_2)
    obj_tot = alpha * obj_1 + (1-alpha)*obj_2
    print('\tTotal objective (alpha*mean_risk + (1-alpha)*expected_excess): ', obj_tot)


def check_and_display(instance, solution, names):
    """Control checker actions"""
    read_solution_from_txt(instance, solution, names)
    penalty, penal_tuple = check_all_constraints(instance)
    # Compute indicators
    mean_risk, quantile = compute_objective(instance)
    alpha = instance[ALPHA_STR]
    q = instance[QUANTILE_STR]
    obj_1 = np.mean(mean_risk)
    tmp = np.zeros(len(quantile))
    obj_2 = np.mean(np.max(np.vstack((quantile - mean_risk, tmp)), axis=0))
    # if penalty > (alpha * obj_1 + (1 - alpha) * obj_2):
        # penalty = (alpha * obj_1 + (1 - alpha) * obj_2)
    obj_tot = alpha * obj_1 + (1 - alpha) * obj_2 + penalty
    return obj_tot, penalty, penal_tuple


def main_checker(instance, solution):
    # instance = read_json('A_08.json')
    intervention_names = list(instance[INTERVENTIONS_STR].keys())
    total, penalty, pen_tup = check_and_display(instance, solution, intervention_names)
    return total, penalty, pen_tup