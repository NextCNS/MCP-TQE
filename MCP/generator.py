import networkx as nx
import numpy as np
import pandas as pd


def generator():
    nodes = [10, 15, 20, 25, 30]
    weight = [0.001, 0.01, 0.05, 0.1]
    for nd in nodes:
        G = nx.erdos_renyi_graph(nd, 6.0/(nd - 1), seed=None, directed=False)
        edges = np.array(G.edges)

        n = len(edges)
        for i in range(0, n):
            edges[i][0] += 1
            edges[i][1] += 1
            edges = np.append(edges, [[edges[i][1], edges[i][0]]], axis=0)
        for w in weight:
            path_w = "output/" + f'net{nd}' + "_" + f'{w}' + ".txt"
            with open(path_w, mode='w') as f:
                f.write(f'{len(G.nodes)} {len(edges)} \n')
                for i in range(0, len(edges)):
                    f.write(' '.join(map(str, edges[i])) + f' {w} \n')


def testlist():
    networks = ["net30"]
    weights = [0.001, 0.01, 0.05, 0.1]
    test = {
        'Filename': [],
        'Weight': [],
        '#Sample': []
    }
    for net in networks:
        for w in weights:
            filename = net + "_" + f'{w}' + "_" + f'{200}' + ".txt"
            test['Filename'].append(filename)
            test['Weight'].append(w)
            test['#Sample'].append(200)

    print(test)
    df = pd.DataFrame({'Filename': test['Filename'],
                       'Weight': test['Weight'],
                       '#Sample': test['#Sample']})

    df.to_csv('testlistex_1.2.csv', index=False)
