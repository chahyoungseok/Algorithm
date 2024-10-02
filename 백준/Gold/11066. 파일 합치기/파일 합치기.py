import sys

T = int(sys.stdin.readline().strip())
INF = sys.maxsize
for _ in range(T):
    K = int(sys.stdin.readline().strip())
    paperSizeList = list(map(int, sys.stdin.readline().strip().split()))

    paperSumList = [0] + [paperSizeList[idx] for idx in range(K)]
    for idx in range(K):
        paperSumList[idx + 1] = paperSumList[idx] + paperSizeList[idx]

    dp = [[INF for _ in range(K)] for _ in range(K)]
    for i in range(K):
        dp[i][i] = 0

    for i in range(1, K):
        for j in range(K - i): # 행
            for k in range(j, j + i):
                dp[j][j + i] = min(
                    dp[j][j + i],
                    dp[j][k] + dp[k + 1][j + i] + paperSumList[j+i+1] - paperSumList[j]
                )

    print(dp[0][K - 1])

# 0 ~ 3 => 0 ~ 1, 2 ~ 3 => k 는 0 ~ j + i - 1까지
# 40 30 30 50
#
# 0   70  160 300
# ~   0   60  170
# ~   ~   0   80
# ~   ~   ~   0
