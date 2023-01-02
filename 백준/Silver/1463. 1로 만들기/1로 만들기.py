N = int(input())
dp = [(x - 1) for x in range(N + 1)]

for i in range(1, N + 1) :
    if i != N :
        dp[i + 1] = min(dp[i] + 1, dp[i + 1])
    if N >= i * 2 :
        dp[i * 2] = min(dp[i] + 1, dp[i * 2])
    if N >= i * 3 :
        dp[i * 3] = min(dp[i] + 1, dp[i * 3])

print(dp[N])