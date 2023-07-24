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

import dimod
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

    upper = m + n - 1
    for j in range(m, m + n):
        # coefficient of t_M

        M = -1
        if gamma[j-m] > 0:
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
            if M >= 0:
                Q[i, upper] += A[j - m, i] * cotM * penalty_two
                Q[upper, i] += A[j - m, i] * cotM * penalty_two

        for i in range(lower, upper):
            Q[i, i] -= pow(pow(2, i - lower), 2) * penalty_two

            Q[i, j] -= pow(2, i - lower) * penalty_two
            Q[j, i] -= pow(2, i - lower) * penalty_two

            for x in range(i + 1, upper):
                Q[i, x] -= pow(2, i - lower) * pow(2, x - lower) * penalty_two
                Q[x, i] -= pow(2, i - lower) * pow(2, x - lower) * penalty_two
            if M >= 0:
                Q[i, upper] -= pow(2, i - lower) * cotM * penalty_two
                Q[upper, i] -= pow(2, i - lower) * cotM * penalty_two

        if M>=0:
            Q[upper, upper] -= pow(cotM, 2) * penalty_two
            Q[j, upper] -= cotM * penalty_two
            Q[upper, j] -= cotM * penalty_two
            # print(Q)
            # print("\n")
    return Q


def QEMC(m, n, k, S, sol, ex,  num_samples = 100):
    # association matrix
    A = np.zeros([n, m], int)
    for i in range(0, n):
        for j in range(0, m):
            if i + 1 in S[j]:
                A[i, j] = 1

    gamma = np.sum(A, axis=1)
    for i in range(0,len(gamma)):
        gamma[i] -= 1
    # num of extra variable
    num_of_ex_var = 0
    for i in gamma:
        if (i > 0):
            num_of_ex_var += int(math.log2(i)) + 1
    print("Extra variables: ", num_of_ex_var)
    # QUBO matrix
    num_of_var = m + n + num_of_ex_var
    Q = np.zeros([num_of_var, num_of_var], int)
    for i in range(m, n + m):
        Q[i, i] = 1

    # Penalty 1
    # penalty_one = max(np.sum(A, axis=0)) * 2 + 5
    penalty_one = 100
    penalty_two = 2
    print("QEMC m, n, p1, p2: ", m, n, penalty_one, penalty_two)
    convert_constraint_1(Q, m, k, penalty_one)
    convert_constraint_2(Q, m, n, gamma, A, penalty_two)
    # Convert MaxCov into a Minimization problem
    Q = -1 * Q
    # Consolidate Q into an upper triangle matrix
    for i in range(0, num_of_var):
        for j in range(i + 1, num_of_var):
            Q[i, j] += Q[j, i]
            Q[j, i] = 0
    non_zeros = np.count_nonzero(Q)
    # Min/max of the diagonal terms
    mindg = min(Q[i, i] for i in range(0, num_of_var))
    maxdg = max(Q[i, i] for i in range(0, num_of_var))
    # Min/max of non-diagonal terms
    minq = min(min(Q[i, i + 1:]) for i in range(0, num_of_var - 1))
    maxq = max(max(Q[i, i + 1:]) for i in range(0, num_of_var - 1))
    print("QEMC Min | Max | Min diagonal | Max diagonal : ", minq, maxq, mindg, maxdg)
    # Run solver
    if ex == 2:
        btime = time.time()
        print("Preparing the QUBO using BINARY formulation...")
        model = dimod.BinaryQuadraticModel.from_qubo(Q, offset=penalty_one * k * k)
        print("--- %s seconds ---" % (time.time() - btime))
        btime = time.time()
        if sol == 'SA':
           sampler = neal.SimulatedAnnealingSampler()
           sampleset = sampler.sample(bqm=model, num_reads=num_samples)

        if sol == 'QA':
            print("Preparing embedding & invoking D-Wave Sampler...")
            DWavesampler = EmbeddingComposite(DWaveSampler())
            chain_strength = int(max(abs(maxq), abs(minq), abs(maxdg), abs(mindg))) + n
            print("Chain strength: ", chain_strength)
            sampleset = DWavesampler.sample(bqm=model, num_reads=num_samples,
                                            return_embedding=True,
                                            chain_strength=chain_strength,
                                            # annealing_time=50
                                            )

            # plot_energies(sampleset, title='Quantum annealing in default parameters')
        print("--- %s seconds ---" % (time.time() - btime))
        btime = time.time()
        print("Processing the solutions...")
        # print(sampleset) to file
        data = sampleset.to_pandas_dataframe()
        output_file = sol + "_BinaryQUBO_samples_" + str(m) + "_" + str(n) + "_" + str(k) + "_" + str(
            num_samples) + ".csv"
        print("Writing samples to file ", output_file)
        data.to_csv(output_file)
        arr = np.array(data, int)
        # print(arr)
        x = arr[:, :m]
        y = arr[:, m:m + n]
        print("Number of samples: ", num_samples)
        # print("Number occurred: ")
        # print(num_oc)
        # print("Chain break: ")
        # print(chain_b)
        coverage = []  # Do not accept violated constraints
        coverage2 = []  # Can violate 2nd constraint, recompute the coverage
        num_violate = 0
        max_co = 0
        qa_violated = 0
        num_oc = arr[:, num_of_var + 1]
        if sol == 'QA':
            num_oc = arr[:, num_of_var + 2]
            chain_b = data[["chain_break_fraction"]].to_numpy()
            print("#Sample(s) with chain breaks: ", np.count_nonzero(chain_b), '/', num_samples)

        for i in range(0, len(arr)):
            tmp = np.zeros(n, int)
            cstr1, cstr2 = verify(x[i], y[i], k, A, sol)
            if not cstr1:
                num_violate += num_oc[i]
            else:
                cov = 0
                for j in range(0, m):
                    if x[i][j] == 1:
                        for z in range(0, n):
                            if (A[z, j] == 1) and (tmp[z] < 0.5):
                                cov += 1
                                tmp[z] = 1
                if cstr2:
                    for tv in range(num_oc[i]):
                        coverage.append(cov)
                for tv in range(num_oc[i]):
                    coverage2.append(cov)
                if cov > max_co:
                    max_co = cov
                    seed = i

        # print(coverage)
        if len(coverage) > 0:
            print("Satisfied BOTH constraints| Max cover: ", max(coverage), " min cover: ", min(coverage), \
                  "avg: ", sum(coverage) * 1.0 / num_samples, " per. feasible: ", len(coverage) * 100.0 / num_samples,
                  "%")
        # print(coverage2)
        if len(coverage2) > 0:
            print("Satisfied FIRST constraint| Max cover: ", max(coverage2), " min cover: ", min(coverage2), \
                  "avg: ", sum(coverage2) * 1.0 / num_samples, " per. feasible: ", len(coverage2) * 100.0 / num_samples,
                  "%")

        # if num_violate == num_samples:
        #     average = 0
        #     min_co = 0
        # else:
        #     average = np.sum(coverage2) / (num_samples - num_violate)
        #     min_co = min(coverage2)
        #     path_w = f'net{m}' + "_" + sol + "_linear.txt"
        #     with open(path_w, mode='w') as f:
        #         for i in range(0, len(x[seed])):
        #             if x[seed, i] > 0.000001:
        #                 f.write(f'{i + 1} \n')
        #
        #     with open('test.csv', mode='w') as f:
        #         for i in range(0, len(x[seed])):
        #             f.write(f'{x[seed][i]} ')
        #             # if x[seed, i] > 0.000001:
        #             #     f.write(f'{i + 1} \n')
        #         f.write('\n')
        #         for j in range(0, len(y[seed])):
        #             f.write(f'{y[seed][j]} ')
        results = {'Max': -1, 'Min': -1, 'average': -1, 'num_of_var': 0, 'non_zeros': 0, 'per_feasible': 0}
        if len(coverage2) > 0:
            results = {
                'Max': max(coverage2),
                'Min': min(coverage2),
                'average': sum(coverage2) * 1.0 / num_samples,
                'num_of_var': num_of_var,
                'non_zeros': non_zeros,
                'per_feasible': len(coverage2) * 100.0 / num_samples
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
