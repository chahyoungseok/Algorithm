import sys

N = int(sys.stdin.readline().strip())

dp = [[0 for _ in range(10)] for _ in range(1001)]
dp[1] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

for i in range(2, 1001) :
    for j in range(10) :
        for k in range(j, 10) :
            dp[i][k] += dp[i - 1][j]

print(sum(dp[N]) % 10007)