N, K = map(int, input().split())

db, dp = [[0,0]], [[0 for _ in range(K + 1)] for _ in range(N + 1)]
for _ in range(N) :
    db.append(list(map(int, input().split())))

for i in range(1, N + 1) :
    for j in range(1, K + 1) :
        weight, value = db[i][0], db[i][1]

        if j < weight :
            dp[i][j] = dp[i - 1][j]
        else :
            dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weight] + value)

print(dp[N][K])