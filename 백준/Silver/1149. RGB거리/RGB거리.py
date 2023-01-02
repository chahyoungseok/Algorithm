import copy, sys

N = int(sys.stdin.readline().strip())
color_cost, dp = [], [[int(1e9), int(1e9), int(1e9)] for _ in range(N)]
for _ in range(N) :
    color_cost.append(list(map(int, (sys.stdin.readline()).split())))

dp[0] = copy.deepcopy(color_cost[0])
for i in range(1, N) :
    for j in range(3) :
        if j == 0 :
            dp[i][j] = min(dp[i - 1][1] + color_cost[i][j], dp[i - 1][2] + color_cost[i][j])
        if j == 1 :
            dp[i][j] = min(dp[i - 1][0] + color_cost[i][j], dp[i - 1][2] + color_cost[i][j])
        if j == 2 :
            dp[i][j] = min(dp[i - 1][0] + color_cost[i][j], dp[i - 1][1] + color_cost[i][j])

print(min(dp[N - 1]))