import sys


standard = 100000
N, K = map(int, (sys.stdin.readline()).split())
if K > N :
    dp = [0 for _ in range(standard + 1)]

    for i in range(N, 0, -1) :
        dp[i - 1] = dp[i] + 1

    for i in range(N, standard) :
        dp[i + 1] = dp[i] + 1

    for i in range(standard + 1) :
        if standard + 1 > i + 1 :
            if dp[i + 1] > dp[i]:
                if dp[i + 1] > dp[i] + 1:
                    dp[i + 1] = dp[i] + 1
            elif dp[i + 1] < dp[i]:
                if dp[i] > dp[i + 1] + 1:
                    dp[i] = dp[i + 1] + 1
        if standard + 1 > i * 2:
            if dp[i * 2] > dp[i]:
                if dp[i * 2] > dp[i]:
                    dp[i * 2] = dp[i]
            elif dp[i * 2] < dp[i]:
                if dp[i] > dp[i * 2]:
                    dp[i] = dp[i * 2]
    print(dp[K])
else :
    print(N - K)
