import sys, heapq

N, M, K = map(int, (sys.stdin.readline()).split())

edges = [[] for _ in range(N + 1)]
for _ in range(M) :
    a, b, c = map(int, (sys.stdin.readline()).split())
    edges[a].append([b, c])
    edges[b].append([a, c])

distances = [[int(1e12) for _ in range(N + 1)] for _ in range(K + 1)]
distances[0][1] = 0
q = [[0, 1, 0]]
while q :
    dist, current_node, wall_break = heapq.heappop(q)

    if current_node == N or dist > distances[wall_break][current_node]:
        continue

    for i in edges[current_node] :
        cost = dist + i[1]
        if distances[wall_break][i[0]] > cost :
            distances[wall_break][i[0]] = cost
            heapq.heappush(q, [cost, i[0], wall_break])
        if K > wall_break and distances[wall_break + 1][i[0]] > dist:
            distances[wall_break + 1][i[0]] = dist
            heapq.heappush(q, [dist, i[0], wall_break + 1])

result = int(1e12)
for i in range(K + 1) :
    result = min(result, distances[i][N])
print(result)