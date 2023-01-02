import copy
import sys


standard = 100000
N, K = map(int, (sys.stdin.readline()).split())
if K > N :
    dp = [[0, [i]] for i in range(standard + 1)]

    for i in range(N, 0, -1) :
        dp[i - 1][0] = dp[i][0] + 1

    for i in range(N, standard) :
        dp[i + 1][0] = dp[i][0] + 1

    for i in range(standard + 1) :
        if standard + 1 > i + 1 :
            if dp[i + 1][0] > dp[i][0]:
                if dp[i + 1][0] > dp[i][0] + 1:
                    dp[i + 1][0] = dp[i][0] + 1
                    dp[i + 1][1] = copy.deepcopy(dp[i][1])
                    dp[i + 1][1].append(i + 1)
            elif dp[i + 1][0] < dp[i][0]:
                if dp[i][0] > dp[i + 1][0] + 1:
                    dp[i][0] = dp[i + 1][0] + 1
                    dp[i][1] = copy.deepcopy(dp[i + 1][1])
                    dp[i][1].append(i)
        if standard + 1 > i * 2:
            if dp[i * 2][0] > dp[i][0]:
                if dp[i * 2][0] > dp[i][0] + 1:
                    dp[i * 2][0] = dp[i][0] + 1
                    dp[i * 2][1] = copy.deepcopy(dp[i][1])
                    dp[i * 2][1].append(i * 2)
            elif dp[i * 2][0] < dp[i][0]:
                if dp[i][0] > dp[i * 2][0] + 1:
                    dp[i][0] = dp[i * 2][0] + 1
                    dp[i][1] = copy.deepcopy(dp[i * 2][1])
                    dp[i][1].append(i)

    print(dp[K][0])
    standard = dp[K][1][0]
    if N > standard :
        for i in range(N, standard, -1) :
            print(i, end=" ")
    elif N < standard :
        for i in range(N, standard, 1) :
            print(i, end=" ")

    for i in dp[K][1] :
        print(i, end=" ")
elif K == N :
    print(0)
    print(str(N))
else :
    print(N - K)
    for i in range(N, K - 1, -1) :
        print(i, end=" ")