from json_reader import read_json
from variable_neighborhood_search import main_vns
from ParticleSwarm import pso_main
from bipop_cmaes import main_bipop
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import random
import time
import copy

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


def main(time_limit, instance_name, output_name, team_id, _):
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

    max_teta = len(Interventions) + len(Exclusions)
    max_teta_2 = pow(len(Interventions), 2) + pow(len(Exclusions), 2)
    penalty_koef_0 = (1 + np.log(max_teta)) * max_teta
    print('penalty koeff', penalty_koef_0)
    #_, _, ini_val, initial, ini_pen = main_vns(instance, initial, (time_limit * 0.02), (1 + 70/len(Interventions)) * 300)
    _, _, ini_val, initial, ini_pen = main_vns(instance, initial, (time_limit * 0.02), penalty_koef_0)
    #_, _, ini_val, initial, ini_pen = main_vns(instance, initial,
    #                                           (time_limit * 0.02), np.log(len(Interventions)+len(Exclusions)))

    #penalty_koef = (ini_val - ini_pen/2) * 50 / max_teta
    # penalty_koef = (1 + 70/len(Interventions)) * 300
    #penalty_koef = np.log(max_teta_2/max_teta)*np.exp(max_teta_2/pow(max_teta, 2))
    penalty_koef = (ini_val - ini_pen/2) * 50 / max_teta
    print('ini_val:', ini_val, 'ini_pen:', ini_pen, 'ini_val-ini_pen:', ini_val - ini_pen, 'max_teta:', max_teta)
    print('penalty koeff', penalty_koef)
    value_cma, penalty_cma, time_cma, best_value_cma, best_penalty_cma, best_solution_cma, ini_population = \
        main_bipop(instance, ini_val, ini_pen, initial, (time_limit * 0.65), penalty_koef)
    population = []
    for i in range(len(ini_population)):
        population.append(list(ini_population[i][0]))

    time_vns, value_vns, best_value_vns, best_solution_vns, best_penalty_vns = \
        main_vns(instance, best_solution_cma, (time_limit * 0.3), penalty_koef)

    f = open(output_name, 'w')
    f.writelines(intervention_names[iteration] + ' ' + str(int(i)) + '\n' for iteration, i in
                 enumerate(best_solution_vns))
    f.close()
    print(team_id, '- Team id ')

    time_cmavns = copy.deepcopy(time_cma)
    t_last = time_cmavns[-1]
    value_cmavns = copy.deepcopy(list(value_cma))
    for i in range(len(time_vns)):
        time_cmavns.append(t_last + i + 1)
        value_cmavns.append(value_vns[i])

    plt.figure(1)
    plt.plot(time_cmavns, value_cmavns, 'b-', label='CMA-VNS')
    plt.legend()
    plt.title("Convergence of solution")
    plt.xlabel("Time")
    plt.ylabel("Objective function")
    plt.grid(True)
    plt.show()
