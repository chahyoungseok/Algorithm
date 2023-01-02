import sys

N, M = map(int, (sys.stdin.readline().split()))
A = [0] + list(map(int, (sys.stdin.readline().split())))
c = [0] + list(map(int, (sys.stdin.readline().split())))

dp = [[0 for _ in range(sum(c) + 1)] for _ in range(N + 1)]
result = int(1e9)

for i in range(1, N + 1) :
    for j in range(1, sum(c) + 1) :
        weight, value = c[i], A[i]

        if j < weight :
            dp[i][j] = dp[i - 1][j]
        else :
            dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weight] + value)

        if dp[i][j] >= M :
            result = min(result, j)

print(result)