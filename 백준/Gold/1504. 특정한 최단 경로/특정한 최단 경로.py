import heapq
import sys


def d(start, end) :
    distance = [int(1e9) for _ in range(N + 1)]
    q, distance[start] = [], 0
    heapq.heappush(q, [0, start])

    while q :
        dist, node = heapq.heappop(q)

        if dist > distance[node] :
            continue

        for i in graph[node] :
            cost = dist + i[0]
            if distance[i[1]] > cost :
                heapq.heappush(q, [cost, i[1]])
                distance[i[1]] = cost

    return distance[end]


N, E = map(int, (sys.stdin.readline()).split())

graph = [[] for _ in range(N + 1)]
for _ in range(E) :
    a, b, c = map(int, (sys.stdin.readline()).split())
    graph[a].append([c, b])
    graph[b].append([c, a])

v1, v2 = map(int, (sys.stdin.readline()).split())

result = min(d(1, v1) + d(v1, v2) + d(v2, N), d(1, v2) + d(v2, v1) + d(v1, N))
if result >= int(1e9) :
    print(-1)
else :
    print(result)