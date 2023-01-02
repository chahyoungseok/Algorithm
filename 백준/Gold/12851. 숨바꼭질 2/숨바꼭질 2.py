import sys


standard = 100000
N, K = map(int, (sys.stdin.readline()).split())
if K > N :
    dp = [[0, 0] for _ in range(standard + 1)]

    for i in range(N, 0, -1) :
        dp[i - 1][0] = dp[i][0] + 1

    for i in range(N, standard) :
        dp[i + 1][0] = dp[i][0] + 1

    for i in range(standard + 1) :
        if standard + 1 > i + 1 :
            if dp[i + 1][0] > dp[i][0]:
                if dp[i + 1][0] > dp[i][0] + 1:
                    dp[i + 1][0] = dp[i][0] + 1
                    dp[i + 1][1] = max(dp[i][1], 1)
                elif dp[i + 1][0] == dp[i][0] + 1:
                    dp[i + 1][1] += max(dp[i][1], 1)
            elif dp[i + 1][0] < dp[i][0]:
                if dp[i][0] > dp[i + 1][0] + 1:
                    dp[i][0] = dp[i + 1][0] + 1
                    dp[i][1] = max(dp[i + 1][1], 1)
                elif dp[i][0] == dp[i + 1][0] + 1:
                    dp[i][1] += max(dp[i + 1][1], 1)
        if standard + 1 > i * 2:
            if dp[i * 2][0] > dp[i][0]:
                if dp[i * 2][0] > dp[i][0] + 1:
                    dp[i * 2][0] = dp[i][0] + 1
                    dp[i * 2][1] = max(dp[i][1], 1)
                elif dp[i * 2][0] == dp[i][0] + 1:
                    dp[i * 2][1] += max(dp[i][1], 1)
            elif dp[i * 2][0] < dp[i][0]:
                if dp[i][0] > dp[i * 2][0] + 1:
                    dp[i][0] = dp[i * 2][0] + 1
                    dp[i][1] = max(dp[i * 2][1], 1)
                elif dp[i][0] == dp[i * 2][0] + 1:
                    dp[i][1] += max(dp[i * 2][1], 1)
    print(dp[K][0])
    print(dp[K][1])
else :
    print(N - K)
    print(1)