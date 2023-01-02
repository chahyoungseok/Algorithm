import sys
from collections import deque


def dfs(edges, current, visited) :
    visited[current] = False
    if current not in dfs_arr :
        dfs_arr.append(current)

    for i in edges[current] :
        if visited[i] :
            dfs(edges, i, visited)


N, M, V = map(int, input().split())

bfs, dfs_arr = [], []
edges = [[] for _ in range(N + 1)]
for _ in range(M) :
    s, e = map(int, (sys.stdin.readline()).split())
    edges[s].append(e)
    edges[e].append(s)

for i in range(1, N + 1) :
    edges[i] = sorted(edges[i])

visited = [True] * (N + 1)
dfs(edges, V, visited)

for i in dfs_arr :
    print(i, end=" ")
print()

q = deque()
q.append(V)
visited = [True] * (N + 1)

while q :
    current = q.popleft()
    if current not in bfs :
        bfs.append(current)
    visited[current] = False

    for i in edges[current] :
        if visited[i] :
            q.append(i)

for i in bfs :
    print(i, end=" ")