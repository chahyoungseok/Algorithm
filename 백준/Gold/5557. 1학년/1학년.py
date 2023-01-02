import sys

N = int((sys.stdin.readline()).strip())
numbers = list(map(int, (sys.stdin.readline()).split()))

dp = [[0 for _ in range(0, 21)] for _ in range(100)]
dp[0][numbers[0]] = 1

for i in range(1, N - 1) :
    target = numbers[i]
    for j in range(21) :
        if 0 <= j + target <= 20 :
            dp[i][j + target] += dp[i - 1][j]
        if 0 <= j - target <= 20 :
            dp[i][j - target] += dp[i - 1][j]

print(dp[N - 2][numbers[N - 1]])