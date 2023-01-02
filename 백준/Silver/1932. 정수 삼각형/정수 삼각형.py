import sys

n = int(sys.stdin.readline().strip())
tri, dp = [], [[0 for _ in range(i + 1)] for i in range(n)]
for _ in range(n) :
    tri.append(list(sys.stdin.readline().split()))

dp[0][0] = int(tri[0][0])
for i in range(1, n) :
    for j in range(i) :
        dp[i][j] = max(dp[i][j], dp[i - 1][j] + int(tri[i][j]))
        dp[i][j + 1] = max(dp[i][j + 1], dp[i - 1][j] + int(tri[i][j + 1]))

print(max(dp[n - 1]))