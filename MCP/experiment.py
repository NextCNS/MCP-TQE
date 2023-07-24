import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def ex_1_1():
    df = pd.read_csv('experiment_1.1.csv')
    linear = []
    binary = []
    nodes = []
    for i in range(0, len(df['Formulation'])):
        if df['Formulation'][i] == 'linear':
            linear.append(df['Embed'][i])
            nodes.append(df['Nodes'][i])
        else:
            binary.append(df['Embed'][i])

    plt.plot(nodes, linear, 'go-', label='Linear')
    plt.plot(nodes, binary, 'ro-', label='Binary')
    plt.title('Experiment 1.1')
    plt.xlabel('Number of nodes')
    plt.ylabel('Number of qubits')
    plt.legend(loc='best')
    plt.show()
    # print(linear)


def ex_1_2():
    df = pd.read_csv('experiment_1.2.csv')
    linear = []
    binary = []
    weight = []
    for i in range(0, len(df['Formulation'])):
        if df['Formulation'][i] == 'linear':
            linear.append(df['Embed'][i])
            weight.append(df['Weight'][i])
        else:
            binary.append(df['Embed'][i])


    # set width of bar
    barWidth = 0.25
    fig = plt.subplots(figsize=(12, 8))

    # Set position of bar on X axis
    br1 = np.arange(len(linear))
    br2 = [x + barWidth for x in br1]

    # Make the plot
    plt.bar(br1, linear, color='g', width=barWidth,
            edgecolor='grey', label='Linear')
    plt.bar(br2, binary, color='r', width=barWidth,
            edgecolor='grey', label='Binary')

    # Adding Xticks
    plt.title('Experiment 1.2')
    plt.xlabel('Weight', fontweight='bold', fontsize=15)
    plt.ylabel('Number of qubits', fontweight='bold', fontsize=15)
    plt.xticks([r + barWidth for r in range(len(linear))],
               [str(w) for w in weight])

    plt.legend()
    plt.show()

