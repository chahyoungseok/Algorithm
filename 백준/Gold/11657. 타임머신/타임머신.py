import sys


def bellman_ford(start) :
    distance[start] = 0

    for i in range(1, N + 1) :
        for j in range(M) :
            now_node, next_node, cost = graph[j]

            if distance[now_node] != int(1e9) and distance[next_node] > distance[now_node] + cost :
                distance[next_node] = distance[now_node] + cost
                if i == N :
                    return True
    return False


N, M = map(int, (sys.stdin.readline()).split())

graph = []
distance = [int(1e9) for _ in range(N + 1)]
for _ in range(M) :
    graph.append(list(map(int, (sys.stdin.readline()).split())))

if bellman_ford(1) :
    print(-1)
else :
    for i in range(2, N + 1) :
        if distance[i] == int(1e9) :
            print(-1)
        else :
            print(distance[i])