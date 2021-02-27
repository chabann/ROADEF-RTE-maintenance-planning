import numpy as np
import numpy.random as rnd
import random
import os
import math
from function_checker import main_checker
import time


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
sign = lambda x: math.copysign(1, x)


def calculateEnergy(x, dis):
    n = len(x)
    energy = 0
    for i in range(n-1):
        energy += dis[x[i]][x[i+1]]
    return energy + dis[x[n-1]][x[0]]


def GenerateStateCandidate(x):
    n = len(x)
    rnd.seed(1000)
    leftbound = random.randint(0, n)
    rightbound = random.randint(0, n)

    if leftbound < rightbound:
        subx0 = x[0: leftbound]
        subx = x[leftbound: rightbound]
        subx1 = x[rightbound: n]
    else:
        subx0 = x[0: rightbound]
        subx = x[rightbound: leftbound]
        subx1 = x[leftbound: n]
    subx = np.flip(subx)
    x1 = np.hstack((subx0, subx, subx1))
    return x1


def GetTransitionProbability(E, t):
    return np.exp(-E/t)


def MakeTransit(p):
    value = rnd.sample()
    if value <= p:
        return 1
    else:
        return 0


def DecreaseTemperature(t, i):
    return t * 0.2 / (i + 1)


def GetStartState(d, delta):
    x = [random.randint(0, d-delta[i]) for i in range(0, d)]
    return x


def sa_main(instance, time_limit, penalty_koef):
    start_time = time.time()
    Interventions = instance[INTERVENTIONS_STR]
    dim = len(Interventions)
    Deltas = []
    intervention_names = list(Interventions.keys())

    for i in range(dim):
        Deltas.append(max(Interventions[intervention_names[i]]['Delta']))

    initialTemperature = 1000000
    endTemperature = 0.0000001
    # iterMax = 15000

    state = [random.randint(1, int(Interventions[intervention_names[i]][TMAX_STR])) for i in range(dim)]
    penalties = []

    currentEnergy, penal, penal_tup = main_checker(instance, state, penalty_koef)
    currentTemp = initialTemperature
    energy = [currentEnergy]
    best_energy = currentEnergy
    best_state = state
    best_penalty = penal
    iter = 0

    while (time.time() - start_time) <= time_limit:
        stateCandidate = GenerateStateCandidate(state)  # получаем состояние - кандидат
        candidateEnergy, penal, penal_tup = main_checker(instance, stateCandidate, penalty_koef)
        penalties.append(penal)

        if candidateEnergy < currentEnergy:
            currentEnergy = candidateEnergy
            state = stateCandidate
            if (candidateEnergy < best_energy) & (penal <= best_penalty):
                best_energy = candidateEnergy
                best_state = stateCandidate
                best_penalty = penal
        else:
            probability = GetTransitionProbability(candidateEnergy - currentEnergy, currentTemp)
            if MakeTransit(probability):
                currentEnergy = candidateEnergy
                state = stateCandidate
        if not (iter % 100):
            energy.append(best_energy)

        currentTemp = DecreaseTemperature(initialTemperature, iter)
        if currentTemp <= endTemperature:
            break
        iter += 1

    print('Done SA. Best solution:', best_energy, 'penalty:', best_penalty)
    time_values = list(range(len(energy)))
    return best_state, best_energy, best_penalty, energy, time_values
