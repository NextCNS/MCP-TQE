import pandas as pd
import numpy as np

import csv

from QEMC import *
from BQMC import *
from generator import *
from experiment import *

if __name__ == "__main__":
    # generator()

    df = pd.read_csv('testlistex2.csv')
    test_list = df.values
    # print(test_list[0])

    formulation = ['linear', 'binary']
    solvers = ['SA', 'QA']
    k = 5
    # collection = {
    #     'Network': [],
    #     'Nodes': [],
    #     'Weight': [],
    #     'Samples': [],
    #     'Formulation': [],
    #     'Variables': [],
    #     'Non-zeros': [],
    #     'Embed': []
    # }
    collection = {
        'Network': [],
        'Weight': [],
        'Samples': [],
        'Solvers': [],
        'Formulation': [],
        'Max': [],
        'Min': [],
        'Average': [],
        'Variables': [],
        'Non-zeros': [],
        '% Feasible': []
    }

    for para in test_list:
        file_name = 'input/' + para[0]
        with open(file_name) as f:
            m, n = [int(x) for x in next(f).split()]  # read first line
            S = []
            for line in f:  # read rest of lines
                array = []
                array.append([int(x) for x in line.split()])
                S.append(array[0])

        f.close()

        ex = 2
        if ex == 1:
            sol = None
            for fo in formulation:
                if fo == 'linear':
                    results = BQMC(m, n, k, S, sol, ex)
                if fo == 'binary':
                    results = QEMC(m, n, k, S, sol, ex)

                collection['Network'].append(para[0])
                collection['Nodes'].append(m)
                collection['Weight'].append(para[1])
                collection['Samples'].append(para[2])
                collection['Formulation'].append(fo)
                collection['Variables'].append(results['num_of_var'])
                collection['Non-zeros'].append(results['non_zeros'])
                collection['Embed'].append(results['embed'])

        if ex == 2:

            for sol in solvers:
                for fo in formulation:
                    if fo == 'linear':
                        results = BQMC(m, n, k, S, sol, ex)
                    if fo == 'binary':
                        results = QEMC(m, n, k, S, sol, ex)

                    collection['Network'].append(para[0])
                    collection['Weight'].append(para[1])
                    collection['Samples'].append(para[2])
                    collection['Solvers'].append(sol)
                    collection['Formulation'].append(fo)
                    collection['Max'].append(results['Max'])
                    collection['Min'].append(results['Min'])
                    collection['Average'].append(results['average'])
                    collection['Variables'].append(results['num_of_var'])
                    collection['Non-zeros'].append(results['non_zeros'])
                    collection['% Feasible'].append(results['per_feasible'])

        df = pd.DataFrame(collection)
        df.to_csv('ex2.csv', index=False)

    df = pd.DataFrame(collection)
    df.to_csv('experiment_2.7.csv', index=False)
    #
    #
    #
    #
    # testlist()
    # ex_1_1()
    #
