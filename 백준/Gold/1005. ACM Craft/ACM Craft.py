import heapq, sys

T = int(input())

for _ in range(T) :
    N, K = map(int, input().split())
    D_list = list(map(str, sys.stdin.readline().split()))

    edges, indeg, min_time = [[] for _ in range(N + 1)], [0] * (N + 1), 0
    for i in range(K) :
        x, y = map(int, (sys.stdin.readline()).split())
        edges[x].append(y)
        indeg[y] += 1

    W = int(input())
    q = []
    for i in range(1, N + 1) :
        if indeg[i] == 0 :
            heapq.heappush(q, [int(D_list[i - 1]), i])
    while q :
        time, node = heapq.heappop(q)
        min_time = time
        if node == W :
            break
        for edge in edges[node] :
            indeg[edge] -= 1
            if indeg[edge] == 0 :
                heapq.heappush(q, [int(D_list[edge - 1]) + int(min_time), edge])

    print(min_time)