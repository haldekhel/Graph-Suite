import networkx as nx
import Queue as que
import numpy as np
import sys
from itertools import combinations

class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0, item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)

def DFS(g, visitedNodes, vertice):
    global dfsOrder
    dfsOrder.append(vertice)
    visitedNodes[vertice] = True

    neighbors = g.neighbors(vertice)
    for w in neighbors:
        if (visited[w] is False):
            DFS(g, visitedNodes, w)

    return dfsOrder

def BFS(g, visitedNodes, vertex):
    global bfsOrder
    global count
    count += 1
    q = Queue()
    visitedNodes[vertex] = True
    q.enqueue(vertex)

    while (q.size() != 0):
        x = q.dequeue()

        bfsOrder.append(x)
        for y in findleast(x):
            if visitedNodes[y] is False:
                count += 1
                visitedNodes[y] = True
                q.enqueue(y)

        count += 1

    return bfsOrder

def MST(D, g, visitedNodes, source):
    global nVertices

    global eT
    global tupl
    global vT
    pq = que.PriorityQueue()
    global tW
    vT.append(source)

    for x, y in g.edges(source):
        p = float(g[x][y]['weight'])
        pq.put((p, (x, y)))

    while pq.empty() is False:
        min = pq.get()
        minEdge = min[1]
        edgeWeight = min[0]

        if (minEdge[1] in vT):
            continue

        if (minEdge[1] not in vT):
            v = minEdge[1]

            for x, y in g.edges(v):
                if (y not in vT):
                    p = float(g[x][y]['weight'])
                    pq.put((p, (x, y)))

        vT.append(v)

        if (minEdge[0] > minEdge[1]):
            eT.append(minEdge[::-1])

        else:
            eT.append(minEdge)

        tW += edgeWeight

    for x,y in eT:
        if (visited[x] is True and visited[y] is True):
            pass

        else:
            p = (x,y,float("{0:.2f}".format(D[x][y])))
            tupl.append(p)

    for x in vT:
        visitedNodes[x] = True

def printMST():
    print("V = " + str(sorted(vT)))

    print("E = ")
    for x in tupl:
        print("     " + str(x))

    print
    print("Total Weight: %.2f" % round(tW,2))
    print


def path(D, S, vi, vj):
    path = list()
    tupl = list()
    path.append(vi)

    x = vi
    tW = 0

    while(x!=vj):
        x = S[x][vj]
        path.append(x)

    for i in range(0, len(path)-1):
            j = i+1
            p = (path[i], path[j], float("{0:.2f}".format(D[path[i]][path[j]])))
            tW += D[path[i]][path[j]]
            tupl.append(p)

    sys.stdout.write(str(vi)+ " -> " + str(vj) + " = ")

    for x in range(len(tupl)):

        if (len(tupl) - 1 == x):
            print(" " + str(tupl[x]))
            break

        sys.stdout.write(str(tupl[x]) + " -> ")

    print("          Path Weight = " + str(tW))

def floyd(g):
    D = [[99999999.0 for i in range(nVertices)] for j in range(nVertices)]
    D = np.array(D)

    S = [[None for i in range(nVertices)] for j in range(nVertices)]
    S = np.array(S)
    M = np.array(nx.to_numpy_matrix(g))

    for i in range(0, nVertices):
        for j in range(0, nVertices):
            if(M[i][j] == 0 and i != j):
                D[i][j] = 99999999.0
            else:
                D[i][j] = M[i][j]
                S[i][j] = j

    for k in range (0, nVertices):
        for i in range (0, nVertices):
            for j in range(0, nVertices):
                if (D[i][j] > D[i][k] + D[k][j]):
                    D[i][j] = D[i][k] + D[k][j]
                    S[i][j] = S[i][k]
    return D, S

def findleast(root):
    neighbors = g.neighbors(root)
    sortedN = list()
    weightedN = {}
    counter = 0

    for x in neighbors:
        counter += 1
        weightedN[x] =  (float(g[root][x]['weight']))

    for y in range(0, counter):
        minNode = min(weightedN, key=weightedN.get)
        del weightedN[minNode]
        sortedN.append(minNode)

    return sortedN


inputData = open(sys.argv[1], 'r')
nVertices = int(inputData.readline())
count = 0
g = nx.Graph()

for line in inputData.readlines():
    line = line.split(" ")
    g.add_edge(int(line[0]), int(line[1]), weight=float(line[2]))

dfsOrder = list()
bfsOrder = list()

visited = [False] * nVertices
print("Depth First Search Traversal: ")

for v in range(len(visited)):
    if(visited[v] is False):
        (DFS(g, visited, v))

print (dfsOrder)
print

visited = [False] * nVertices
print("Breadth First Search Traversal: ")

for v in range(len(visited)):
    if (visited[v] is False):
        (BFS(g, visited, v))


print(bfsOrder)
print

D, S = floyd(g)

visited = [False] * nVertices
eT = list()
tupl = list()
vT = list()
tW = 0
for v in range(len(visited)):
    if (visited[v] is False):
        MST(D,g,visited,v)
printMST()

# print("Shortest Paths: ")
# for x in combinations((g.nodes()), 2):
#     path(np.array(D), S, x[0], x[1])

connectedComponents = list()
for x in nx.connected_components(g):
    connectedComponents.append(sorted(list(x)))

for x in connectedComponents:
    for y in combinations(x, 2):
        path(D, S, y[0], y[1])