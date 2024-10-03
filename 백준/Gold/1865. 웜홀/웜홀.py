import sys


def bellman_ford(start):
    distances = [INF for _ in range(N + 1)]
    distances[start] = 0

    for i in range(1, N + 1):
        for now_node, next_node, time in edges :
            if distances[next_node] > distances[now_node] + time:
                distances[next_node] = distances[now_node] + time
                if i == N :
                    return False
    return True


TC = int(sys.stdin.readline().strip())
INF = sys.maxsize
for _ in range(TC):
    N, M, W = map(int, sys.stdin.readline().strip().split())
    edges = []

    for _ in range(M):
        S, E, T = map(int, sys.stdin.readline().strip().split())
        edges.append([S, E, T])
        edges.append([E, S, T])

    for _ in range(W):
        S, E, T = map(int, sys.stdin.readline().strip().split())
        edges.append([S, E, -T])

    if bellman_ford(1):
        print("NO")
    else:
        print("YES")