import sys, heapq

V, E = map(int, (sys.stdin.readline()).split())
start = int(sys.stdin.readline().strip())

edges = [[] for _ in range(V + 1)]
distances = [int(1e9) for _ in range(V + 1)]

for _ in range(E) :
    u, v, w = map(int, (sys.stdin.readline()).split())
    edges[u].append([v, w])

q = [[0, start]]
distances[start] = 0

while q :
    dist, node = heapq.heappop(q)

    if dist > distances[node] :
        continue

    for i in edges[node] :
        cost = i[1] + dist
        if distances[i[0]] > cost :
            distances[i[0]] = cost
            heapq.heappush(q, [cost, i[0]])

for i in range(1, V + 1) :
    if distances[i] == int(1e9) :
        print("INF")
    else :
        print(distances[i])