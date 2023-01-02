import sys
from collections import deque

N = int(sys.stdin.readline().strip())
LENGTH = 21
graph = [[] for _ in range(N + 1)]
parent = [[0 for _ in range(LENGTH)] for _ in range(N + 1)]
visited = [True for _ in range(N + 1)]
depth = [0 for _ in range(N + 1)]

for _ in range(N - 1) :
    a, b = map(int, (sys.stdin.readline()).split())
    graph[a].append(b)
    graph[b].append(a)

q = deque()
q.append([1, 0])
depth[1], visited[1] = 0, False

while q :
    node, dist = q.popleft()

    for i in graph[node] :
        if visited[i]:
            parent[i][0], visited[i] = node, False
            depth[i] = dist + 1
            q.append([i, dist + 1])

for i in range(1, LENGTH):
    for j in range(1, N + 1):
        parent[j][i] = parent[parent[j][i - 1]][i - 1]

M = int(sys.stdin.readline().strip())
for _ in range(M) :
    a, b = map(int, (sys.stdin.readline()).split())

    if depth[a] > depth[b] :
        a, b = b, a

    for i in range(LENGTH - 1, -1, -1):
        if depth[b] - depth[a] >= 2 ** i:
            b = parent[b][i]

    if a == b :
        print(a)
        continue

    for i in range(LENGTH - 1, -1, -1):
        if parent[a][i] != parent[b][i]:
            a = parent[a][i]
            b = parent[b][i]

    print(parent[a][0])