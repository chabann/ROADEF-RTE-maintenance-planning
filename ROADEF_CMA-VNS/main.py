from json_reader import read_json
from variable_neighborhood_search import main_vns
from bipop_cmaes import main_bipop
from SAfuncSolver import sa_main
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

value_as, solution_as, penalty_as = main_as(instance, initial)

# value_cma, penalty_cma, time_cma, best_value_cma, best_penalty_cma, best_solution_cma = main_bipop(instance)
value_cma, penalty_cma, time_cma, best_value_cma, best_penalty_cma, best_solution_cma = main_bipop(instance, initial)

time_vns, value_vns, best_value_vns, best_solution_vns, best_penalty_vns = main_vns(instance, best_solution_cma)

best_state_sa, best_value_sa, best_penalty_sa, value_sa, time_sa = sa_main(instance)

# value_as, solution_as, penalty_as = main_as(instance, best_solution_cma)

plt.figure(1)
plt.plot(time_cma, value_cma, 'b-')
plt.grid()

plt.figure(2)
plt.plot(time_vns, value_vns, 'b-')
plt.grid()

time_cmavns = time_cma
t_last = time_cmavns[-1]
value_cmavns = list(value_cma)
for i in range(len(time_vns)):
    time_cmavns.append(t_last + i + 1)
    value_cmavns.append(value_vns[i])

time_total = max(time_cmavns, time_sa)
if time_cmavns < time_total:
    for i in range(len(time_cmavns), len(time_total)):
        time_cmavns.append(i)
        value_cmavns.append(value_cmavns[-1])

if time_sa < time_total:
    for i in range(len(time_sa), len(time_total)):
        time_sa.append(i)
        value_sa.append(value_sa[-1])

plt.figure(3)
plt.plot(time_cmavns, value_cmavns, 'b-')
plt.grid()

plt.figure(4)
plt.plot(time_total, value_cmavns, 'b-', label='CMA-VNS')
plt.plot(time_total, value_sa, 'r-', label='SA')
plt.grid()

plt.show()

f = open('output/' + instance_name + ".txt", 'w')

f.writelines(intervention_names[iter] + ' ' + str(int(i)) + '\n' for iter, i in enumerate(best_solution_vns))
f.close()

