from json_reader import read_json
from adaptive_search import main_as
import os
import matplotlib.pyplot as plt
import random

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

instance_name = 'A_09'
instance = read_json(instance_name + '.json')
Interventions = instance[INTERVENTIONS_STR]

dim = len(Interventions)
Deltas = []
Resources = instance[RESOURCES_STR]
Exclusions = instance[EXCLUSIONS_STR]
Time = instance[T_STR]

intervention_names = list(Interventions.keys())

for i in range(dim):
    Deltas.append(max(Interventions[intervention_names[i]]['Delta']))

initial = [random.randint(1, int(Interventions[intervention_names[i]][TMAX_STR])) for i in range(dim)]
value_as, solution_as, penalty_as = main_as(instance)

