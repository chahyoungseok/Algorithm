import sys

n = int(sys.stdin.readline().strip())
dp = [0] * 1001
dp[1], dp[2] = 1, 3

for i in range(3, 1001) :
    dp[i] = (dp[i - 1] + dp[i - 2] + dp[i - 2]) % 10007

print(dp[n])