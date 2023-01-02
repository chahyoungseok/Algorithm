n = int(input())
dp = [0] * 1001
dp[1], dp[2] = 1, 2
for i in range(3, 1001) :
    dp[i] = dp[i - 1] + dp[i - 2]

print(dp[n] % 10007)