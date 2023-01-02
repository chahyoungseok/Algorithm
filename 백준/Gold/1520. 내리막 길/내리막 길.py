import sys
from collections import deque

N, M = map(int, input().split())

board, total = [], 1
for _ in range(N) :
    board.append(list(map(int, (sys.stdin.readline()).split())))

dx, dy = [1, -1, 0, 0], [0, 0, -1, 1]
dp = [[0 for _ in range(M)] for _ in range(N)]

q, dp[0][0] = deque(), 1
q.append([0, 0])

while q :
    x, y = q.popleft()

    if x == N - 1 and y == M - 1:
        continue

    for i in range(4) :
        mx, my = x + dx[i], y + dy[i]
        if mx >= 0 and mx < N and my >= 0 and my < M and board[x][y] > board[mx][my] :
            dp[mx][my] += dp[x][y]

            if [mx, my] not in q :
                q.append([mx, my])
    dp[x][y] = 0
    
print(dp[N - 1][M - 1])
