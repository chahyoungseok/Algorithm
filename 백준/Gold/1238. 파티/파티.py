import sys, heapq

N, M, X = map(int, (sys.stdin.readline()).split())

edges, max_distance = [[] for _ in range(N + 1)], 0
for _ in range(M) :
    s, e, t = map(int, (sys.stdin.readline()).split())
    edges[s].append([e, t])


def dijkstra(start) :
    distances = [int(1e9) for _ in range(N + 1)]
    distances[start] = 0

    q = []
    heapq.heappush(q, [0, start])
    while q :
        dist, node = heapq.heappop(q)

        if dist > distances[node] :
            continue

        for i in edges[node] :
            cost = dist + i[1]
            if distances[i[0]] > cost :
                heapq.heappush(q, [cost, i[0]])
                distances[i[0]] = cost

    return distances


edge_distances = [[int(1e9) for _ in range(N + 1)]]
for i in range(1, N + 1) :
    edge_distances.append(dijkstra(i))

for i in range(1, N + 1) :
    total_distance = edge_distances[i][X] + edge_distances[X][i]
    if int(1e9) > total_distance :
        max_distance = max(max_distance, total_distance)
print(max_distance)
