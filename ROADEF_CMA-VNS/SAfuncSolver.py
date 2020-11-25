import numpy as np
import numpy.random as rnd
import random
import os
import math
from function_checker import main_checker


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


def GenerateStateCandidate(x, t, delta):
    d = len(x)
    rnd.seed(1000)
    leftbound = random.randint(0, d-1)
    rightbound = random.randint(0, d-1)
    if leftbound > rightbound:
        leftbound, rightbound = rightbound, leftbound
    for i in range(leftbound, rightbound):
        x[i] = 0
        while x[i] == 0:
            alpha = random.random()
            # alpha = (sign(alpha - 0.5)*t*((1 - 1/t)**(2*alpha - 1))*time).real
            alpha = (sign(alpha - 0.5) * t * ((1 + d/t) ** (2 * alpha - 1) - 1)).real
            if (alpha >= 0) & (alpha <= d-delta[i]):
                x[i] = round(alpha)
    return x


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


def sa_main(instance):
    Interventions = instance[INTERVENTIONS_STR]
    dim = len(Interventions)
    Deltas = []
    intervention_names = list(Interventions.keys())

    for i in range(dim):
        Deltas.append(max(Interventions[intervention_names[i]]['Delta']))

    initialTemperature = 100000
    endTemperature = 0.000001
    iterMax = 15000

    state = GetStartState(dim, Deltas)  # задаём вектор начального состояния, как случайную перестановку работ
    penalties = []

    currentEnergy, penal, penal_tup = main_checker(instance, state)
    currentTemp = initialTemperature
    energy = [currentEnergy]
    best_energy = currentEnergy
    best_state = state
    best_penalty = penal
    best_pen_tup = penal_tup
    time = [1]

    for iter in range(iterMax):
        stateCandidate = GenerateStateCandidate(state, currentTemp, Deltas)  # получаем состояние - кандидат
        candidateEnergy, penal, penal_tup = main_checker(instance, stateCandidate)
        penalties.append(penal)

        if candidateEnergy < currentEnergy:
            currentEnergy = candidateEnergy
            state = stateCandidate
            if (candidateEnergy < best_energy) & (penal <= best_penalty):
                best_energy = candidateEnergy
                best_state = stateCandidate
                best_penalty = penal
                best_pen_tup = penal_tup
        else:
            probability = GetTransitionProbability(candidateEnergy - currentEnergy, currentTemp)
            if MakeTransit(probability):
                currentEnergy = candidateEnergy
                state = stateCandidate
        if not (iter % 100):
            energy.append(best_energy)
            time.append(time[-1])

        currentTemp = DecreaseTemperature(initialTemperature, iter)  # уменьшаем температуру

        if currentTemp <= endTemperature:
            break

    print('SA Solver has finished, best energy:', best_energy, 'best penalty:', best_penalty)
    return best_state, best_energy, best_penalty, energy, time
