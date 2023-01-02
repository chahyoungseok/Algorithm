import sys

N = int(sys.stdin.readline().strip())
dp = [i for i in range(N + 1)]
square_number = [i ** 2 for i in range(1, int(N ** 0.5) + 1)]

for i in range(1, N + 1) :
    for j in square_number :
        if j > i:
            break
        if dp[i] > dp[i - j] + 1 :
            dp[i] = dp[i - j] + 1

print(dp[N])
