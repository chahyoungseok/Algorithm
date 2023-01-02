import sys

N = int(sys.stdin.readline().strip())

board, dp = [], [[0 for _ in range(N)] for _ in range(N)]
for _ in range(N) :
    board.append(list(map(int, (sys.stdin.readline()).split())))

dp[0][0] = 1
q = [[] for _ in range(N * 2 - 2)]
q[0].append([0, 0, board[0][0]])

for i in range(N * 2 - 2) :
    for x, y, jump in q[i] :
        if N > x + jump :
            jump_x = board[x + jump][y]
            if (x + jump == N - 1 and y == N - 1) or jump_x != 0 :
                dp[x + jump][y] += dp[x][y]
                if N * 2 - 2 > x + y + jump and [x + jump, y, jump_x] not in q[x + y + jump]:
                    q[x + y + jump].append([x + jump, y, jump_x])

        if N > y + jump :
            jump_y = board[x][y + jump]
            if (x == N - 1 and y + jump == N - 1) or jump_y != 0 :
                dp[x][y + jump] += dp[x][y]
                if N * 2 - 2 > x + y + jump and [x, y + jump, jump_y] not in q[x + y + jump]:
                    q[x + y + jump].append([x, y + jump, jump_y])

print(dp[N - 1][N - 1])