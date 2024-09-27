import sys

N, M = map(int, sys.stdin.readline().strip().split())

country = [[0 for _ in range(M + 1)]]
for _ in range(N) :
    country.append([0] + list(map(int, sys.stdin.readline().strip().split())))

for i in range(N):
    for j in range(M + 1):
        country[i + 1][j] += country[i][j]

for i in range(N + 1):
    for j in range(M):
        country[i][j + 1] += country[i][j]


K = int(sys.stdin.readline().strip())
for _ in range(K):
    x1, y1, x2, y2 = map(int, sys.stdin.readline().strip().split())

    print(country[x2][y2] + country[x1 - 1][y1 - 1] - country[x1 - 1][y2] - country[x2][y1 - 1])