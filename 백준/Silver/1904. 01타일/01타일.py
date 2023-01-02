import sys

N = int(sys.stdin.readline().strip())
dp = [0] * 1000001
dp[0], dp[1], dp[2] = 0, 1, 2

for i in range(3, 1000001) :
    dp[i] = (dp[i - 1] + dp[i - 2]) % 15746
print(dp[N])