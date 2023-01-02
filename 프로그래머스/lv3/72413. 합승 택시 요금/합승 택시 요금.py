def solution(n, s, a, b, fares):
    min_distance = int(1e9)
    graph = [[int(1e9)] * (n + 1) for _ in range(n + 1)]
    for fare in fares:
        c, d, f = fare
        graph[c][d] = f
        graph[d][c] = f

    for i in range(1, n + 1):
        graph[i][i] = 0

    for k in range(1, n + 1):
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                graph[i][j] = min(graph[i][j], graph[i][k] + graph[k][j])

    for i in range(1, n + 1) :
        min_distance = min(graph[s][i] + graph[i][a] + graph[i][b], min_distance)

    return min_distance