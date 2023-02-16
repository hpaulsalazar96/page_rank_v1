import numpy as np
import pandas as pd
import random as rd
import networkx as nx
import matplotlib.pyplot as plt
from tabulate import tabulate


def graph_my_graph(stochasticMatrix, numpyStochasticMatrix):
    nodeNameList = stochasticMatrix.columns.values
    matrixSize = stochasticMatrix.__len__()

    graph = nx.DiGraph()
    graph.add_nodes_from(nodeNameList)

    numpySMLen = len(numpyStochasticMatrix)
    print("MATRIX SIZE: " + str(matrixSize))

    for indexNode in range(numpySMLen):
        tempList = numpyStochasticMatrix[indexNode]
        for indexTempNodeVal in range(len(tempList)):
            listTempValues = tempList[indexTempNodeVal]
            if listTempValues != 0:
                graph.add_edge("N" + str(indexTempNodeVal + 1), "N" + str(indexNode + 1))

    nx.draw_networkx(graph, with_labels=True, node_color='lightgreen', arrows=True, arrowsize=8)
    plt.draw()
    plt.title("15 Initial Node Graph")
    plt.show()


def ask(trap):
    userInput = input(trap + ", TELEPORT? Y/N:  ")
    if userInput == "y" or userInput == "Y":
        return True
    else:
        return False


def page_rank(stochasticMatrix, numpyStochasticMatrix, beta):
    matrixSize = stochasticMatrix.__len__()
    maxValIndex = matrixSize - 1
    rt = np.transpose(np.ones(matrixSize) * 1 / matrixSize)
    pt = np.transpose(np.zeros(matrixSize))
    pt[maxValIndex] = 0.5
    visitedNodes = [maxValIndex + 1]
    returnList = []

    flag_trap = 0
    counter = 1

    while len(set(visitedNodes)) < 15:
        print("ACTUAL MATRIX")
        print("ACTUAL NODE: " + str(maxValIndex + 1))
        head_ = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
        print(tabulate(numpyStochasticMatrix, showindex=head_, headers=head_, tablefmt='orgtbl'))
        np.set_printoptions(linewidth=np.inf)
        print("Vector rt: ")
        print(rt)
        print("Vector pt: ")
        print(pt)
        print("VISITED NODES: ")
        print(visitedNodes)
        print(str(counter) + " RUNNING TIME")

        rt = np.round(np.dot(numpyStochasticMatrix, rt), 2)
        pt = np.round(np.dot(numpyStochasticMatrix, pt), 2)
        returnList.append(np.count_nonzero(rt == min(rt)))
        maxValIndex = np.argmax(pt)
        visitedNodes.append((maxValIndex + 1))

        if sum(stochasticMatrix.loc[:, "N" + str(maxValIndex + 1)]) == 0 \
                or list(set(stochasticMatrix.loc[:, "N" + str(maxValIndex + 1)])).__len__() == 1:
            if ask('DEAD-END TRAP'):
                numpyStochasticMatrix[:, maxValIndex] = np.ones(matrixSize) * 1 / matrixSize
                newIndex = rd.randrange(0, matrixSize)
                while newIndex == maxValIndex and visitedNodes.__contains__(newIndex):
                    newIndex = rd.randrange(0, matrixSize)
                numpyStochasticMatrix[newIndex, maxValIndex] = 0.9
                flag_trap = 1

        if flag_trap == 0 and returnList.__len__() >= 3 and \
                (returnList[returnList.__len__() - 3] - returnList[returnList.__len__() - 1]) == 0 and \
                returnList[returnList.__len__() - 3] > 0:
            if ask("SPYDER TRAP"):
                returnList = []
                nodeWeight = np.asarray(np.ones((matrixSize, matrixSize))) * 1 / matrixSize
                numpyStochasticMatrix = np.array(
                    (np.asarray(numpyStochasticMatrix) * beta) + ((1 - beta) * nodeWeight))
                newIndex = rd.randrange(0, matrixSize)
                while newIndex == maxValIndex and visitedNodes.__contains__(newIndex):
                    newIndex = rd.randrange(0, matrixSize)
                numpyStochasticMatrix[newIndex, maxValIndex] = 0.9

        flag_trap = 0
        counter += 1

    # TODO calculate new stochastic matrix after falling into a trap

    print(list(set(visitedNodes)))

    # pagerank = []
    # for index, val in enumerate(rt):
    #     pagerank.append((index + 1, val))
    # pagerank.sort(key=lambda a: a[1], reverse=True)
    # print(pagerank)


if __name__ == "__main__":
    betaArg = 0.8
    stochasticMatrixArg = pd.read_excel('AdjacencyMatrix.xlsx', index_col=0)
    numpyStochasticMatrixByArg = stochasticMatrixArg.to_numpy()
    print(stochasticMatrixArg)

    graph_my_graph(stochasticMatrixArg, numpyStochasticMatrixByArg)
    page_rank(stochasticMatrixArg, numpyStochasticMatrixByArg, betaArg)
