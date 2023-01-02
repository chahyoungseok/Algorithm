import sys
from collections import deque


def check(node) :
    q = deque()
    q.append(node)
    visited = [True for _ in range(N)]
    visited[node] = False
    while q :
        n = q.popleft()

        for e in edges[n] :
            next_node, cost = e
            if next_node == E :
                return True
            if visited[next_node] :
                visited[next_node] = False
                q.append(next_node)

    return False


def bellman_ford(start) :
    distance[start] = input_cost[S]

    for i in range(N) :
        for j in range(N) :
            for k in edges[j] :
                now_node = j
                next_node, cost = k
                i_cost = input_cost[next_node]

                if distance[now_node] != -int(1e12) and distance[now_node] - cost + i_cost > distance[next_node]:
                    distance[next_node] = distance[now_node] - cost + i_cost
                    if i == N - 1 :
                        if check(next_node) :
                            return True
    return False


N, S, E, M = map(int, (sys.stdin.readline()).split())
edges = [[] for _ in range(N)]
distance = [-int(1e12) for _ in range(N)]

for _ in range(M) :
    start, end, value = map(int, (sys.stdin.readline()).split())
    edges[start].append([end, value])
input_cost = list(map(int, (sys.stdin.readline()).split()))

result = bellman_ford(S)
if distance[E] == -int(1e12) :
    print("gg")
elif result :
    print("Gee")
else :
    print(distance[E])