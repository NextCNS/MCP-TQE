import networkx as nx
import minorminer
from dwave_networkx.generators.pegasus import pegasus_graph
import sys


def embed(Q):
    G = nx.Graph()
    s = 16  # generate p_s graph to embed
    time_lim = 3600  # the embeding stop after time_lim seccond

    for i in range(0, len(Q) - 1):
        for j in range(i + 1, len(Q)):
            if Q[i, j] != 0:
                G.add_edge(i, j)
    connectivity_structure = pegasus_graph(s)
    embedded_graph = minorminer.find_embedding(G.edges(), connectivity_structure.edges(), threads=10, timeout=time_lim)
    cnt = 0
    for x in embedded_graph:
        cnt = cnt + len(embedded_graph[x])
    print(str(cnt))
    return cnt
