from function_checker import main_checker
from json_reader import read_json
from variable_neighborhood_search import main_vns
from bipop_cmaes import main_bipop

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

instance_name = 'A_09'
instance = read_json(instance_name + '.json')
Interventions = instance[INTERVENTIONS_STR]

dim = len(Interventions)
print('Interventions: ', dim)
Deltas = []

Resources = instance[RESOURCES_STR]
print('Resources: ', len(Resources))

Exclusions = instance[EXCLUSIONS_STR]
print('Exclusions: ', len(Exclusions))

Time = instance[T_STR]
print('T: ', Time)

intervention_names = list(Interventions.keys())

for i in range(dim):
    Deltas.append(max(Interventions[intervention_names[i]]['Delta']))

current_value, current_penalty, current_time, best_value, best_penalty, best_solution = main_bipop(instance)

time, value, best_cost, best_solution, best_penalty = main_vns(instance, best_solution)

plt.figure(1)
plt.plot(current_time, current_value, 'b-')
plt.grid()

plt.figure(2)
plt.plot(time, value, 'b-')
plt.grid()

time_tot = current_time
t_last = time_tot[-1]
value_tot = current_value
for i in range(len(time)):
    time_tot.append(t_last + i + 1)
    value_tot.append(value[i])

plt.figure(3)
plt.plot(time_tot, value_tot, 'b-')
plt.grid()

plt.show()

f = open('output/' + instance_name + ".txt", 'w')

f.writelines(intervention_names[iter] + ' ' + str(int(i)) + '\n' for iter, i in enumerate(best_solution))
f.close()

