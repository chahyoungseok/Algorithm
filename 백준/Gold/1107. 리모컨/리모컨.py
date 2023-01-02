N = int(input())
M = int(input())
r, dp = [], [int(1e9)] * 1000000
if M != 0 :
    r = list(map(str, input().split()))

if not r :
    if abs(N - 100) > 2 :
        print(len(str(N)))
    else :
        print(abs(N - 100))

else :
    for i in range(1000000) :
        state = True
        for j in str(i) :
            if j in r :
                state = False
                break

        if state :
            dp[i] = len(str(i))

    dp[100] = 0
    for i in range(1, 1000000) :
        dp[i] = min(dp[i - 1] + 1, dp[i])

    for i in range(999999, -1, -1) :
        dp[i - 1] = min(dp[i] + 1, dp[i - 1])

    print(dp[N])
