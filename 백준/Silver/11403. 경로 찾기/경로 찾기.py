import sys

N = int(sys.stdin.readline().strip())

graph = [[0 for _ in range(N + 1)]]
for _ in range(N) :
    graph.append([0] + list(map(int, sys.stdin.readline().strip().split())))

for k in range(1, N + 1):
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            if graph[i][k] == 1 and graph[k][j] == 1:
                graph[i][j] = 1

for i in range(1, N + 1):
    for j in range(1, N + 1):
        print(graph[i][j], end=" ")
    print()