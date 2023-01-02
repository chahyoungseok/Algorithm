S = int(input())
dp = [int(1e9)] * 1001
dp[0], dp[1] = 1, 0
for i in range(1, 1001) :
    index = 1
    for j in range(i + i, 1001, i) :
        dp[j] = min(dp[j], dp[i] + index + 1)
        index += 1

        sel = 1
        for k in range(j - 1, 0, -1) :
            dp[k] = min(dp[k], dp[j] + sel)
            sel += 1

print(dp[S])