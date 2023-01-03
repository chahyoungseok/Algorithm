import sys, heapq

N, M = map(int, (sys.stdin.readline()).split())

edges = [[] for _ in range(N + 1)]
for _ in range(M) :
    A, B, C = map(int, (sys.stdin.readline()).split())
    edges[A].append([C, B])
    edges[B].append([C, A])


for i in range(N) :
    edges[i] = sorted(edges[i], reverse=True)

start, end = map(int, (sys.stdin.readline()).split())

q = [[-int(1e9), start]]
dp = [0 for _ in range(N + 1)]

while q :
    value, node = heapq.heappop(q)
    value = -value

    if node == end :
        print(value)
        break

    if dp[node] > value :
        continue

    for c, r in edges[node] :
        cost = min(value, c)
        if cost > dp[r] :
            heapq.heappush(q, [-cost, r])
            dp[r] = cost
