import heapq

INF = int(1e9)
visitCount = 0
maxDistance = 0

N, M, C = map(int, input().split())

graph = [[] for _ in range(N + 1)]

for i in range(M) :
    X, Y, Z = map(int, input().split())
    graph[X].append((Y,Z))

distances = [INF for _ in range(N + 1)]

q = []
heapq.heappush(q, (0,C))
distances[C] = 0

while q :
    dist, now = heapq.heappop(q)

    if dist > distances[now] :
        continue

    for i in graph[now] :
        cost = dist + i[1]
        if distances[i[0]] > cost :
            distances[i[0]] = cost
            heapq.heappush(q, (cost, i[0]))

for i in range(1, M + 1) :
    if distances[i] != INF :
        visitCount += 1
        maxDistance = max(maxDistance, distances[i])

print("가장 멀리있는 노드의 거리는 : " + str(maxDistance) + "\n방문한 노드의 개수는 : " + str(visitCount))