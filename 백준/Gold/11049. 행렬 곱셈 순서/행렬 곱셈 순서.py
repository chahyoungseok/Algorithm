import sys

N = int(sys.stdin.readline().strip())

INF = sys.maxsize
rc_list = []
for _ in range(N):
    rc_list.append(list(map(int, sys.stdin.readline().strip().split())))

dp = [[INF for _ in range(N)] for _ in range(N)]
for i in range(N):
    dp[i][i] = 0

for i in range(1, N): # 대각선
    for j in range(N - i): # 행
        for k in range(i + j):
            dp[j][i + j] = min(
                dp[j][i + j],
                dp[j][k] + dp[k + 1][i + j] + (rc_list[j][0] * rc_list[k][1] * rc_list[i + j][1])
            )

print(dp[0][N - 1])

# j 0 ~ N - i, k 0 ~ 3 -> k = 2 dp[i][k] + dp[k][i + j]
# dp[0][3] = min(dp[0][1] + dp[1][3] + 합치는 비용, dp[0][2] + dp[2][3] + 합치는 비용)
# dp[ABC] = min(dp[A] + dp[BC] + 합치는 비용, dp[AB] + dp[C] + 합치는 비용)