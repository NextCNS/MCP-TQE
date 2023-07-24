#
# ================================================================================================
import pandas as pd
import numpy as np
import math
from verify import *
from embed import *
import dimod

import neal
from dwave.system import DWaveSampler, EmbeddingComposite
import time
import matplotlib.pyplot as plt
from collections import Counter



def convert_constraint_1(Q, m, k, penalty_one):
    # input: number of sets, number of elements, parameter k, number of binary variables for constraint 1
    # penalty value
    # output: matrix Q

    for i in range(0, m):
        # linear term
        Q[i, i] = - (1 - 2 * k) * penalty_one

        # quadratic term
        for j in range(i + 1, m):
            Q[i, j] = Q[j, i] = - penalty_one

    return Q


def convert_constraint_2(Q, m, n, gamma, A, penalty_two):
    # input: number of sets, number of elements, number of binary variables for constraint 2
    # output: matrix Q
    # constraint 2

    upper = m + n
    for j in range(m, m + n):

        M = int(gamma[j - m])-1

        lower = upper
        upper = lower + M
        Q[j, j] -= penalty_two
        for i in range(0, m):
            # linear term
            Q[i, i] -= A[j - m, i] * penalty_two

            # quadratic term
            for x in range(i + 1, m):
                Q[i, x] -= A[j - m, i] * A[j - m, x] * penalty_two
                Q[x, i] -= A[j - m, i] * A[j - m, x] * penalty_two
            Q[i, j] += A[j - m, i] * penalty_two
            Q[j, i] += A[j - m, i] * penalty_two

            for x in range(lower, upper):
                Q[i, x] += A[j - m, i] * penalty_two
                Q[x, i] += A[j - m, i] * penalty_two

        for i in range(lower, upper):
            Q[i, i] -= penalty_two

            Q[i, j] -= penalty_two
            Q[j, i] -= penalty_two

            for x in range(i + 1, upper):
                Q[i, x] -= penalty_two
                Q[x, i] -= penalty_two

        # print(Q)
        # print("\n")
    return Q


def convert_constraint_3(Q, m, n, gamma, penalty_two):
    upper = n + m
    for x in range(0, m):
        lower = upper
        upper = lower + gamma[x]
        for i in range(lower, upper - 1):
            Q[i + 1, i + 1] += 1
            Q[i + 1, i] -= penalty_two
            Q[i, i + 1] -= penalty_two

    return Q


def plot_energies(results, title=None):
    energies = results.data_vectors['energy']
    occurrences = results.data_vectors['num_occurrences']
    counts = Counter(energies)
    total = sum(occurrences)
    counts = {}
    for index, energy in enumerate(energies):
        if energy in counts.keys():
            counts[energy] += occurrences[index]
        else:
            counts[energy] = occurrences[index]
    for key in counts:
        counts[key] /= total
    df = pd.DataFrame.from_dict(counts, orient='index').sort_index()
    df.plot(kind='bar', legend=None)

    plt.xlabel('Energy')
    plt.ylabel('Probabilities')
    plt.title(str(title))
    plt.show()
    print("minimum energy:", min(energies))

def CQM(m, n, k, S, sol, ex, num_samples = 100):
    A = np.zeros([m, n], int)
    for i in range(0, m):
        for j in S[i]:
            A[i, j-1] = 1
    print(m, n)
    print(A)
    # Add variables x_i, i = 1..m
    x = [dimod.Binary(f'x_{i}') for i in range(0, m)]
    # Add variables y_j, j = 1..n
    y = [dimod.Binary(f'y_{j}') for j in range(0, n)]
    # Add objective
    cqm = dimod.ConstrainedQuadraticModel()
    cqm.set_objective(-sum(y))
    # Add constraint \sum_i x_i = k
    cqm.add_constraint(sum(x) - k <= 0, label='cardinality')
    # Add constraints \sum_i A[i, j ]*x_i - y_j \geq 0, j = 1..n
    for j in range(0, n):
        cqm.add_constraint(y[j] - sum(A[i,j] * x[i] for i in range(0,m)) <= 0, label=f'cover y_{j}' )
    # Solve using LeapHybridCQMSampler
    from dwave.system import LeapHybridCQMSampler
    sampler = LeapHybridCQMSampler()
    sampleset = sampler.sample_cqm(cqm)
    # print(sampleset) to file
    data = sampleset.to_pandas_dataframe()
    output_file = sol + "_BQMC_samples_" + str(m) + "_" + str(n) + "_" + str(k) + "_" + str(num_samples) + ".csv"
    print("Writing samples to file ", output_file)
    data.to_csv(output_file)


    # results = {
    #     'Max': max(coverage2),
    #     'Min': min(coverage2),
    #     'average': sum(coverage2)*1.0/num_samples,
    #     'num_of_var': num_of_var,
    #     'non_zeros': non_zeros,
    #     'per_feasible': len(coverage2)*100.0/num_samples
    # }



    return {}
