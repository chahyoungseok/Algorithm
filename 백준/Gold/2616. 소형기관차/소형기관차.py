import sys

N = int(sys.stdin.readline().strip())
train = list(map(int, (sys.stdin.readline().split())))
K = int(sys.stdin.readline().strip())

people_sum, sum_value = [], sum(train[0 : K])
people_sum.append(sum_value)
for i in range(1, N - K + 1) :
    sum_value -= train[i - 1]
    sum_value += train[i + K - 1]
    people_sum.append(sum_value)

dp = [[0 for _ in range(N)] for _ in range(3)]
for i in range(1, N - K + 2) :
    dp[0][i] = max(dp[0][i - 1], people_sum[i - 1])

for i in range(1, 3) :
    for j in range(1 + (i * K), N - K + 2) :
        dp[i][j] = max(dp[i][j - 1], dp[i - 1][j - K] + people_sum[j - 1])

print(max(dp[2]))