import sys

N, M = map(int, sys.stdin.readline().strip().split())

cumulativeSum = [[0 for _ in range(M + 1)]]
for _ in range(N):
    cumulativeSum.append([0] + list(map(int, sys.stdin.readline().strip().split())))

for i in range(N):
    for j in range(M + 1):
        cumulativeSum[i + 1][j] = cumulativeSum[i][j] + cumulativeSum[i + 1][j]

for i in range(N + 1):
    for j in range(M):
        cumulativeSum[i][j + 1] = cumulativeSum[i][j] + cumulativeSum[i][j + 1]


K = int(sys.stdin.readline().strip())
for _ in range(K):
    i, j, x, y = map(int, sys.stdin.readline().strip().split())
    print(cumulativeSum[x][y] - cumulativeSum[i - 1][y] - cumulativeSum[x][j - 1] + cumulativeSum[i - 1][j - 1])

