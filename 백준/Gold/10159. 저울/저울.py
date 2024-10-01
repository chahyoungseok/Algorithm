import sys

N = int(sys.stdin.readline().strip())
M = int(sys.stdin.readline().strip())

INF = sys.maxsize
graph = [[INF for _ in range(N + 1)] for _ in range(N + 1)]
for _ in range(M):
    a, b = map(int, sys.stdin.readline().strip().split())
    graph[a][b] = 1
    graph[b][a] = -1

for i in range(N + 1):
    graph[i][i] = 0

for k in range(1, N + 1):
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            if graph[i][k] == 1 and graph[k][j] == 1:
                graph[i][j] = 1
            if graph[i][k] == -1 and graph[k][j] == -1:
                graph[i][j] = -1

for i in range(1, N + 1):
    edge_sum = N
    for j in range(1, N + 1):
        if graph[i][j] != INF:
            edge_sum -= 1
    print(edge_sum)

