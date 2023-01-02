import heapq

T = int(input())
dx = [-1, 0, 1, 0]
dy = [0, 1, 0, -1]
result = []

for _ in range(T) :
    N = int(input())

    graph = []
    distance = [[int(1e9)] * N for _ in range(N)]

    for _ in range(N) :
        graph.append(list(map(int, input().split())))

    q = []
    distance[0][0] = graph[0][0]
    heapq.heappush(q, (graph[0][0], 0, 0))

    while q:
        dist, nx, ny = heapq.heappop(q)

        if dist > distance[nx][ny]:
            continue

        for i in range(4):
            cx = nx + dx[i]
            cy = ny + dy[i]
            if cx < 0 or cx >= N or cy < 0 or cy >= N:
                continue

            cost = dist + graph[cx][cy]
            if distance[cx][cy] > cost:
                distance[cx][cy] = cost
                heapq.heappush(q, (cost, cx, cy))

    result.append(distance[N - 1][N - 1])

print(result)