import sys

N = int(input())
l, dp = [], [0] * N
for _ in range(N) :
    l.append(int(sys.stdin.readline()))

if 3 > N :
    print(sum(l))
else :
    dp[0] = l[0]
    dp[1] = l[0] + l[1]
    dp[2] = max(l[0] + l[2], l[1] + l[2])
    for i in range(3, N) :
        dp[i] = max(dp[i - 3] + l[i] + l[i - 1], dp[i - 2] + l[i])

    print(dp[N - 1])