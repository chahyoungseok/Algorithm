import sys

N = int(sys.stdin.readline().strip())
N_list = list(map(int, (sys.stdin.readline()).split()))

dp = [[0 for _ in range(N)] for _ in range(N)]
for i in range(N) :
    dp[i][i] = 1

for i in range(1, N) :
    for j in range(i) :
        # dp[i][j] 판단
        # N_list[j] ~ N_list[i]를 보고 판단
        if N_list[i] == N_list[j]:
            if i - j == 1 :
                dp[i][j] = 1
            elif dp[i - 1][j + 1] == 1 :
                dp[i][j] = 1

M = int(sys.stdin.readline().strip())
for _ in range(M) :
    S, E = map(int, (sys.stdin.readline()).split())
    print(dp[E - 1][S - 1])
