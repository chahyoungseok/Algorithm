import sys

N = int(sys.stdin.readline().strip())

dp = [0] * (N + 2)
for i in range(1, N + 1) :
    T, P = map(int, (sys.stdin.readline()).split())
    dp[i] = max(dp[i], dp[i - 1])
    if N + 2 > i + T :
        dp[i + T] = max(dp[i + T], dp[i] + P)

print(max(dp))