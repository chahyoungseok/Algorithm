import sys
from collections import deque


def bfs(node, graph, visited) :
    q = deque()
    q.append(node)
    sum_node = 0

    while q :
        current_node = q.popleft()
        for n in graph[current_node] :
            if visited[n] :
                visited[n] = False
                q.append(n)
                sum_node += 1

    return sum_node


N, M = map(int, (sys.stdin.readline()).split())
count = 0

graph_d = [[] for _ in range(N + 1)]
graph_u = [[] for _ in range(N + 1)]

for _ in range(M) :
    # a가 b보다 작다.
    a, b = map(int, (sys.stdin.readline()).split())
    graph_d[b].append(a)
    graph_u[a].append(b)

for i in range(1, N + 1) :
    visited_u, visited_d = [True for _ in range(N + 1)], [True for _ in range(N + 1)]
    visited_u[i], visited_d[i] = False, False

    if bfs(i, graph_d, visited_d) + bfs(i, graph_u, visited_u) == N - 1:
        count += 1

print(count)