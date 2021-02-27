from json_reader import read_json
from variable_neighborhood_search import main_vns
from bipop_cmaes import main_bipop
from SAfuncSolver import sa_main
from adaptive_search import main_as
from ParticleSwarm import pso_main
from datetime import datetime
import seaborn as sns

import os
import matplotlib.pyplot as plt
import random
import copy
import time

large = 34; med = 22; small = 16
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("white")

# Global variables
start_time = time.time()
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
ALPHA_STR = "Alpha"


def main_all(time_limit, instance_name, _, team_id, seed):
    time_start = time.time()
    instance = read_json(instance_name + '.json')
    Interventions = instance[INTERVENTIONS_STR]
    Exclusions = instance[EXCLUSIONS_STR]
    dim = len(Interventions)
    Deltas = []
    intervention_names = list(Interventions.keys())

    for i in range(dim):
        Deltas.append(max(Interventions[intervention_names[i]]['Delta']))

    initial = [random.randint(1, int(Interventions[intervention_names[i]][TMAX_STR])) for i in range(dim)]
    time_limit = time_limit - (time.time() - time_start)

    _, _, ini_val, initial, ini_pen = main_vns(instance, initial,
                                               (time_limit * 0.05), (1 + 70 / len(Interventions)) * 300)

    max_teta = len(Interventions) + len(Exclusions)
    penalty_koef = (ini_val - ini_pen) * 50 / max_teta
    print('penalty koeff', penalty_koef)
    #value_cma, penalty_cma, time_cma, best_value_cma, best_penalty_cma, best_solution_cma, ini_population = \
    #    main_bipop(instance, initial, (time_limit * 0.25), penalty_koef)
    #population = []
    #for i in range(len(ini_population)):
    #    population.append(list(ini_population[i][0]))

    #time_vns, value_vns, best_value_vns, best_solution_vns, best_penalty_vns = \
    #    main_vns(instance, best_solution_cma, (time_limit * 0.25), penalty_koef)

    #time_pso_c, values_pso_c, value_pso_c, solution_pso_c, penalty_pso_c = \
    #    pso_main(instance, best_solution_cma, (time_limit*0.25), penalty_koef, population)

    best_state_sa, best_value_sa, best_penalty_sa, value_sa, time_sa = sa_main(instance,
                                                                               (time_limit*0.25), penalty_koef)
    
    time_pso, values_pso, value_pso, solution_pso, penalty_pso = pso_main(instance, initial,
                                                                          (time_limit*0.25), penalty_koef)

    #time_cmavns = copy.deepcopy(time_cma)
    #t_last = time_cmavns[-1]
    #value_cmavns = copy.deepcopy(list(value_cma))
    #for i in range(len(time_vns)):
    #    time_cmavns.append(t_last + i + 1)
    #    value_cmavns.append(value_vns[i])

    #time_cmapso = copy.deepcopy(time_cma)
    #t_last = time_cmapso[-1]
    #value_cmapso = copy.deepcopy(list(value_cma))
    #for i in range(len(time_pso_c)):
    #    time_cmapso.append(t_last + i + 1)
    #    value_cmapso.append(values_pso_c[i])

    #time_total = list(range(max(len(time_cmavns), len(time_cmapso), len(time_pso), len(time_sa))))

    #if len(time_cmavns) < len(time_total):
    #    for i in range(len(time_cmavns), len(time_total)):
    #        value_cmavns.append(value_cmavns[-1])

    #if len(time_sa) < len(time_total):
    #    for i in range(len(time_sa), len(time_total)):
    #        value_sa.append(value_sa[-1])

    #if len(time_cmapso) < len(time_total):
    #    for i in range(len(time_cmapso), len(time_total)):
    #        value_cmapso.append(value_cmapso[-1])

    #if len(time_pso) < len(time_total):
    #    for i in range(len(time_pso), len(time_total)):
    #        values_pso.append(values_pso[-1])

    #plt.figure(1)
    #plt.plot(time_total, value_cmavns, 'b-', label='CMA-VNS')
    #plt.plot(time_total, value_cmapso, 'r-', label='CMA-PSO')
    #plt.plot(time_total, value_sa, 'g-', label='SA')
    #plt.plot(time_total, values_pso, 'y-', label='PSO')
    #plt.legend()
    #plt.title("Comparative values")
    #plt.xlabel("Time")
    #plt.ylabel("Objective function")
    #plt.grid(True)
    #plt.show()
