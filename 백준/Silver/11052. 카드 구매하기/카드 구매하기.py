import sys

N = int(sys.stdin.readline().strip())
P_list = [0] + list(map(int, (sys.stdin.readline()).split()))

dp = [0] * (N + 1)
dp[1] = P_list[1]
for i in range(N) :
    for j in range(1, N + 1) :
        if N + 1 > i + j :
            dp[i + j] = max(dp[i + j], dp[i] + P_list[j])

print(dp[N])