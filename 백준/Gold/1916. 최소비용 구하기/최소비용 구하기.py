import heapq
import sys

N = int(input())
M = int(input())

edges = [[] for _ in range(N + 1)]
distances = [int(1e9) for _ in range(N + 1)]
for _ in range(M) :
    a, b, c = map(int, (sys.stdin.readline()).split())
    edges[a].append([c, b])

A, B = map(int, input().split())

q = []
heapq.heappush(q, [0, A])
distances[A] = 0

while q :
    dist, node = heapq.heappop(q)

    if dist > distances[node] :
        continue

    for i in edges[node] :
        cost = dist + i[0]
        if distances[i[1]] > cost :
            distances[i[1]] = cost
            heapq.heappush(q, [cost, i[1]])

print(distances[B])