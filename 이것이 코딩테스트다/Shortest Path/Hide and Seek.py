import heapq

N,  M = map(int, input().split())

graph = [[] for _ in range(N + 1)]
for _ in range(M) :
    A, B = map(int, input().split())
    graph[A].append((B, 1))
    graph[B].append((A, 1))

distances = [int(1e9)] * (N + 1)

def dijkstra(start) :
    q = []
    heapq.heappush(q, (0, start))
    distances[start] = 0

    while q :
        dist, now = heapq.heappop(q)

        if dist > distances[now] :
            continue

        for i in graph[now] :
            cost = dist + i[1]
            if distances[i[0]] > cost :
                distances[i[0]] = cost
                heapq.heappush(q, (cost, i[0]))

    return distances

distance_case = dijkstra(1)
distance_case.remove(int(1e9))
max_sel, max_index, max_same = distance_case[0], 0, 0

for i in range(1, N) :
    if distance_case[i] > max_sel :
        max_sel = distance_case[i]
        max_index = i

for i in range(N) :
    if distance_case[i] == max_sel :
        max_same += 1

print(str(max_index + 1) + " " + str(max_sel) + " " + str(max_same))