N, M = map(int, input().split())
result_sum = 0
result = [0] * (N + 1)

graph = [[int(1e9)] * (N + 1) for _ in range(N + 1)]

for i in range(N + 1) :
    graph[i][i] = 0

for _ in range(M) :
    A, B = map(int, input().split())
    graph[A][B] = 1

for k in range(1, N + 1) :
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            if graph[i][k] == 1 and graph[k][j] == 1 :
                graph[i][j] = 1


for i in range(1, N + 1) :
    for j in range(1, N + 1) :
        if graph[i][j] == 1 :
            result[i] += 1
            result[j] += 1

for i in result :
    if i == 5 :
        result_sum += 1

print(result_sum)