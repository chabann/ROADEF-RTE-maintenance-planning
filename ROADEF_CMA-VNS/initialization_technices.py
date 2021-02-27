from function_checker import main_checker
from json_reader import read_json
from function_check_constr import checker_constraints
from function_check_value import check_value
import random
import numpy as np


instance = read_json('A_10' + '.json')
TMAX_STR = 'tmax'
INTERVENTIONS_STR = 'Interventions'
EXCLUSIONS_STR = 'Exclusions'
T_STR = 'T'

Exclusions = instance[EXCLUSIONS_STR]
Time = instance[T_STR]
Interventions = instance[INTERVENTIONS_STR]
intervention_names = list(Interventions.keys())
dim = len(Interventions)


initial = [random.randint(1, int(Interventions[intervention_names[i]][TMAX_STR])) for i in range(dim)]
value, penalty, _ = main_checker(instance, initial)
print('value1:', value, 'penalty1:', penalty)

# ---------------------------------------------------------------------#

L = [1 for _ in range(dim)]

U = [round(int(Interventions[intervention_names[i]][TMAX_STR])) for i in range(dim)]

gamma = round(Time // 5)
I = [round((U[i] - L[i])/gamma) for i in range(dim)]

S = np.zeros((gamma, dim))
S[0] = L

for beta in range(1, gamma-1):
    S[beta] = S[beta - 1] + I

S[gamma-1] = U

D_ar = list(range(dim))
x = np.array(())
x = np.append(x, S[0])
solution = []
Omega = [0 for _ in range(gamma)]
i = 0
for beta in range(gamma):
    for j in reversed(D_ar):
        for alpha in range(gamma):
            if (beta == alpha) and (Omega[beta] == 0):
                Omega[beta] = 1
                x = np.vstack((x, S[beta]))
                x[i, j] = S[alpha, j]
                i += 1
            else:
                if (beta == alpha) and (Omega[beta] == 1):
                    continue
                else:
                    x = np.vstack((x, S[beta]))
                    x[i, j] = S[alpha, j]
                    i += 1

max_teta = 2 * len(Interventions) + len(Exclusions)
tau = max_teta / 2
PS = len(x)
Teta = []
for z in range(PS):
    Teta.append(checker_constraints(instance, x[z]))

n_g = 5
PS_g = []
x_g = {}
Teta_g = {}
f_g = {}
f_g_min = np.inf
x_g_min = []

for g in range(n_g):
    x_g[g] = []
    Teta_g[g] = []
    f_g[g] = []
    if g != (n_g - 1):
        for i in range(round(PS // n_g)):
            if Teta[i] < tau:
                x_g[g].append(list(x[i]))
                Teta_g[g].append(Teta[i])
                f_g[g].append(check_value(instance, x_g[g][-1]))
                if f_g[g][-1] < f_g_min:
                    f_g_min = f_g[g][-1]
                    x_g_min = x_g[g][-1]
    else:
        for i in range(PS - round(PS // n_g) * (n_g - 1)):
            if Teta[i] < tau:
                x_g[g].append(list(x[i]))
                Teta_g[g].append(Teta[i])
                f_g[g].append(check_value(instance, x_g[g][-1]))
                if f_g[g][-1] < f_g_min:
                    f_g_min = f_g[g][-1]
                    x_g_min = x_g[g][-1]

    PS_g.append(len(x_g[g]))

"""f_ = [np.sum([f_g[g][i] for i in range(PS_g[g])]) / PS_g[g] for g in range(n_g)]

lambda_f = sum([sum([f_[gf] for gf in range(n_g)])/f_[g] for g in range(n_g)])

Index_f_g = [(sum([f_[gf] for gf in range(n_g)])/f_[g]) * lambda_f**(-1) for g in range(n_g)]

Teta__g = [max(sum([Teta_g[g][i] for i in range(PS_g[g])])/PS_g[g], 0.0001) for g in range(n_g)]

lambda_vio = sum([sum([Teta__g[gv] for gv in range(n_g)])/Teta__g[g] for g in range(n_g)])

Index_teta_g = [(sum([Teta__g[gv] for gv in range(n_g)])/Teta__g[g]) * lambda_vio**(-1) for g in range(n_g)]

AvgIndex_g = [(Index_f_g[g] + Index_teta_g[g])/2 for g in range(n_g)]

PS_g_fin = [AvgIndex_g[g]/sum(AvgIndex_g)*(round(PS // 3)) for g in range(n_g)]"""


value, penalty, _ = main_checker(instance, x_g_min)
print('value2:', value, 'penalty2:', penalty)
