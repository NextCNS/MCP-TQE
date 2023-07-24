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

        M = int(gamma[j - m])

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
        for i in range(lower, upper-1):
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



def BQMC(m, n, k, S, sol, ex, num_samples = 100):
    # association matrix
    EPS = 0.000001
    A = np.zeros([n, m], int)
    for i in range(0, n):
        for j in range(0, m):
            if i + 1 in S[j]:
                A[i, j] = 1

    gamma = np.sum(A, axis=1)
    for i in range(0, len(gamma)):
        gamma[i] -= 1
    num_of_extra_variable = np.sum(gamma)

    # QUBO matrix
    num_of_var = m + n + num_of_extra_variable
    Q = np.zeros([num_of_var, num_of_var], int)
    for i in range(m, n + m):
        Q[i, i] = 1

    # Penalty 1
    #penalty_one = max(np.sum(A, axis=0)) * 2 + 5
    penalty_one = 100
    penalty_two = 2
    print("BQMC m, n, p1, p2: ", m, n, penalty_one, penalty_two)
    Q = convert_constraint_1(Q, m, k, penalty_one)

    Q = convert_constraint_2(Q, m, n, gamma, A, penalty_two)
    #Q = convert_constraint_3(Q, m, n, gamma, penalty_two)
    # Convert MaxCov into a Minimization problem
    Q = -1 * Q
    #Consolidate Q into an upper triangle matrix
    for i in range(0, num_of_var):
        for j in range(i+1, num_of_var):
            Q[i, j] += Q[j, i]
            Q[j, i]  = 0
    non_zeros = np.count_nonzero(Q)
    # Min/max of the diagonal terms
    mindg = min(Q[i, i] for i in range(0, num_of_var))
    maxdg = max(Q[i, i] for i in range(0, num_of_var))
    # Min/max of non-diagonal terms
    minq  = min(min(Q[i,i+1:]) for i in range(0, num_of_var-1))
    maxq  = max(max(Q[i,i+1:]) for i in range(0, num_of_var-1))
    print("BQMC Min| Max | Min diagonal | Max diagonal : ", minq, maxq, mindg, maxdg)

    if ex == 2:
        btime = time.time()
        print("Preparing the QUBO using LINEAR formulation...")
        model = dimod.BinaryQuadraticModel.from_qubo(Q, offset=penalty_one * k * k)
        print("--- %s seconds ---" % (time.time() - btime))
        btime = time.time()
        if sol == 'SA':
            sampler = neal.SimulatedAnnealingSampler()
            sampleset = sampler.sample(bqm=model, num_reads=num_samples)
            # sampler.sample_qubo( Q, num_reads=num_samples)
        if sol == 'QA':
            # linear = {}
            # quadratic = {}
            # for i in range(0, num_of_var):
            #     key = (f'{i}', f'{i}')
            #     linear.setdefault(key, Q[i, i])
            #     for j in range(i + 1, num_of_var):
            #         if abs(Q[i, j] + Q[j, i]) > 0.00000001:
            #             key = (f'{i}', f'{j}')
            #             quadratic.setdefault(key, Q[i, j] + Q[j, i])
            # print(linear)
            # print(quadratic)
            # matrix = {**linear, **quadratic}
            print("Preparing embedding & invoking D-Wave Sampler...")
            # sampler_auto = EmbeddingComposite(DWaveSampler(solver={'topology__type': 'pegasus'}))
            # sampleset = sampler_auto.sample_qubo(matrix,
            #                                    #  chain_strength = 1.4*max(map(max, Q)),
            #                                    #  annealing_time = 50,
            #                                    #  chain_strength = 450,
            #                                      offset=,
            #                                      num_reads=num_samples)
            DWavesampler = EmbeddingComposite(DWaveSampler())
            chain_strength  = int(max( abs(maxq), abs(minq), abs(maxdg), abs(mindg)))+n
            print("Chain strength: ", chain_strength)
            sampleset = DWavesampler.sample(bqm=model, num_reads=num_samples,
                                               return_embedding=True,
                                               chain_strength=chain_strength,
                                               # annealing_time=50
                                               )

            #plot_energies(sampleset, title='Quantum annealing in default parameters')
        print("--- %s seconds ---" % (time.time() - btime))
        btime = time.time()
        print("Processing the solutions...")
        #print(sampleset) to file
        data = sampleset.to_pandas_dataframe()
        output_file = sol+"_LinearQUBO_samples_" + str(m) + "_" + str(n) + "_" + str(k) +"_" +str(num_samples)+".csv"
        print("Writing samples to file ", output_file)
        data.to_csv(output_file)
        arr = np.array(data, int)
        #print(arr)
        x = arr[:, :m]
        y = arr[:, m:m + n]
        print("Number of samples: ", num_samples)
        #print("Number occurred: ")
        #print(num_oc)
        #print("Chain break: ")
        #print(chain_b)
        coverage = []    #Do not accept violated constraints
        coverage2 = []   #Can violate 2nd constraint, recompute the coverage
        num_violate = 0
        max_co = 0
        qa_violated = 0
        num_oc  = arr[:, num_of_var + 1]
        if sol == 'QA':
            num_oc = arr[:, num_of_var + 2]
            chain_b = data[["chain_break_fraction"]].to_numpy()
            print("#Sample(s) with chain breaks: ", np.count_nonzero(chain_b),'/',num_samples)

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



        #print(coverage)
        if len(coverage) >0:
            print("Satisfied BOTH constraints| Max cover: ", max(coverage), " min cover: ",min(coverage),\
                  "avg: ", sum(coverage)*1.0/num_samples ," per. feasible: ", len(coverage)*100.0/num_samples,"%")
        #print(coverage2)
        if len(coverage2) >0:
            print("Satisfied FIRST constraint| Max cover: ", max(coverage2), " min cover: ",min(coverage2),\
                  "avg: ", sum(coverage2)*1.0/num_samples ," per. feasible: ", len(coverage2)*100.0/num_samples,"%")

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
        results = {'Max': -1, 'Min':-1, 'average':-1, 'num_of_var': 0, 'non_zeros':0, 'per_feasible':0}
        if len(coverage2) > 0:
            results = {
                'Max': max(coverage2),
                'Min': min(coverage2),
                'average': sum(coverage2)*1.0/num_samples,
                'num_of_var': num_of_var,
                'non_zeros': non_zeros,
                'per_feasible': len(coverage2)*100.0/num_samples
            }

    if ex == 1:
        em = embed(Q)
        results = {
            'num_of_var': num_of_var,
            'non_zeros': non_zeros,
            'embed': em
        }

    print("--- %s seconds ---" % (time.time() - btime))
    btime = time.time()
    print(results)
    return results
