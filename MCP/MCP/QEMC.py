#
# ================================================================================================

import numpy as np
import math
import pandas as pd
import csv
from verify import *
from embed import *
import neal
from dwave.system import DWaveSampler, EmbeddingComposite


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

    upper = m + n - 1
    for j in range(m, m + n):
        # coefficient of t_M
        M = int(math.log2(gamma[j - m]))
        cotM = gamma[j - m] + 1 - pow(2, M)

        lower = upper + 1
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
                Q[i, x] += A[j - m, i] * pow(2, x - lower) * penalty_two
                Q[x, i] += A[j - m, i] * pow(2, x - lower) * penalty_two

            Q[i, upper] += A[j - m, i] * cotM * penalty_two
            Q[upper, i] += A[j - m, i] * cotM * penalty_two

        for i in range(lower, upper):
            Q[i, i] -= pow(pow(2, i - lower), 2) * penalty_two

            Q[i, j] -= pow(2, i - lower) * penalty_two
            Q[j, i] -= pow(2, i - lower) * penalty_two

            for x in range(i + 1, upper):
                Q[i, x] -= pow(2, i - lower) * pow(2, x - lower) * penalty_two
                Q[x, i] -= pow(2, i - lower) * pow(2, x - lower) * penalty_two

            Q[i, upper] -= pow(2, i - lower) * cotM * penalty_two
            Q[upper, i] -= pow(2, i - lower) * cotM * penalty_two

        Q[upper, upper] -= pow(cotM, 2) * penalty_two

        Q[j, upper] -= cotM * penalty_two
        Q[upper, j] -= cotM * penalty_two
        # print(Q)
        # print("\n")
    return Q


def QEMC(m, n, k, S, sol, ex):

    # association matrix
    A = np.zeros([n, m], int)
    for i in range(0, n):
        for j in range(0, m):
            if i + 1 in S[j]:
                A[i, j] = 1

    gamma = np.sum(A, axis=1)

    # num of extra variable
    num_of_ex_var = 0

    for i in gamma:
        num_of_ex_var += int(math.log2(i)) + 1

    # QUBO matrix
    num_of_var = m + n + num_of_ex_var
    Q = np.zeros([num_of_var, num_of_var], int)
    for i in range(m, n + m):
        Q[i, i] = 1

    # Penalty 1
    penalty_one = max(np.sum(A, axis=0)) * 2 + 5
    convert_constraint_1(Q, m, k, penalty_one)

    penalty_two = 3
    convert_constraint_2(Q, m, n, gamma, A, penalty_two)
    non_zeros = 0
    for i in range(0, num_of_var):
        for j in range(0, num_of_var):
            if abs(Q[i, j] + Q[j, i]) > 0.0000001:
                non_zeros += 1

    if ex == 2:
        num_samples = 1000
        if sol == 'SA':
            sampler = neal.SimulatedAnnealingSampler()
            sampleset = sampler.sample_qubo(-1 * Q, num_reads=num_samples)

        if sol == 'QA':
            linear = {}
            quadratic = {}
            Q = -1 * Q
            for i in range(0, num_of_var):
                key = (f'{i}', f'{i}')
                linear.setdefault(key, Q[i, i])
                for j in range(i + 1, num_of_var):
                    if abs(Q[i, j] + Q[j, i]) > 0.00000001:
                        key = (f'{i}', f'{j}')
                        quadratic.setdefault(key, Q[i, j] + Q[j, i])
            matrix = {**linear, **quadratic}
            sampler_auto = EmbeddingComposite(DWaveSampler(solver={'topology__type': 'pegasus'}))
            sampleset = sampler_auto.sample_qubo(matrix, num_reads=num_samples)

        data = sampleset.to_pandas_dataframe()
        arr = np.array(data, int)

        x = arr[:, :m]
        y = arr[:, m:m + n]

        coverage = []
        num_violate = 0
        max_co = 0
        for i in range(0, len(arr)):
            tmp = A
            if not verify(x[i], y[i], k, A, sol):
                num_violate += 1
            else:
                if sol == 'SA':
                    coverage.append(np.sum(y[i]))
                    if np.sum(y[i]) > max_co:
                        max_co = np.sum(y[i])
                        seed = i
                else:
                    cov = 0
                    for j in range(0, len(x[i])):
                        for z in range(0, len(y[i])):
                            if x[i][j] == 1:
                                if tmp[z, j] == 1:
                                    cov += tmp[z, j]
                                    tmp[z, :] = 0
                    coverage.append(cov)
                    if cov > max_co:
                        max_co = cov
                        seed = i

        per_feasible = (num_samples - num_violate) / num_samples * 100

        if num_violate == num_samples:
            average = 0
            min_co = 0
        else:
            average = np.sum(coverage) / (num_samples - num_violate)
            min_co = min(coverage)
            path_w = f'net{m}' + "_" + sol + "_binary.seeds"
            with open(path_w, mode='w') as f:
                for i in range(0, len(x[seed])):
                    if x[seed, i] > 0.000001:
                        f.write(f'{i + 1} \n')

            with open('test.csv', mode='w') as f:
                for i in range(0, len(x[seed])):
                    f.write(f'{x[seed][i]} ')
                    # if x[seed, i] > 0.000001:
                    #     f.write(f'{i + 1} \n')
                f.write('\n')
                for j in range(0, len(y[seed])):
                    f.write(f'{y[seed][j]} ')

        results = {
            'Max': max_co,
            'Min': min_co,
            'average': average,
            'num_of_var': num_of_var,
            'non_zeros': non_zeros,
            'per_feasible': per_feasible
        }

    if ex == 1:
        em = embed(Q)
        results = {
            'num_of_var': num_of_var,
            'non_zeros': non_zeros,
            'embed': em
        }

    print(results)
    return results
