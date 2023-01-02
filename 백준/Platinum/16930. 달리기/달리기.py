import sys
from collections import deque

N, M, K = map(int, (sys.stdin.readline()).split())

board = []
for _ in range(N) :
    board.append(list(sys.stdin.readline().strip()))

x1, y1, x2, y2 = map(int, (sys.stdin.readline()).split())
x1, y1, x2, y2 = x1 - 1, y1 - 1, x2 - 1, y2 - 1
dx, dy = [1, -1, 0, 0], [0, 0, 1, -1]

dp = [[int(1e9) for _ in range(M)] for _ in range(N)]
q, state, dp[x1][y1] = deque(), True, 0
q.append([x1, y1])

while q and state:
    x, y = q.popleft()

    for i in range(4) :
        mx, my, k = x + dx[i], y + dy[i], 1
        while 0 <= mx < N and 0 <= my < M and board[mx][my] == '.' and K >= k and dp[mx][my] >= dp[x][y] + 1:
            if mx == x2 and my == y2 :
                print(dp[x][y] + 1)
                state = False
                break

            if dp[mx][my] == int(1e9) :
                q.append([mx, my])
                dp[mx][my] = dp[x][y] + 1

            k += 1
            mx += dx[i]
            my += dy[i]

        if not state :
            break

if state :
    print(-1)

