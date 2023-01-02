import sys
from collections import deque

K = int(sys.stdin.readline().strip())
W, H = map(int, (sys.stdin.readline()).split())

board, result = [], -1
for _ in range(H) :
    board.append(list(map(int, (sys.stdin.readline()).split())))

dx, dy = [1, -1, 0, 0, -2, -1, 1, 2, 2, 1, -1, -2], [0, 0, 1, -1, 1, 2, 2, 1, -1, -2, -2, -1]

dp = [[[0 for _ in range(K + 1)] for _ in range(W)] for _ in range(H)]

q = deque()
q.append([0, 0, 0])

while q :
    x, y, k = q.popleft()

    if x == H - 1 and y == W - 1 :
        result = dp[x][y][k]
        break

    for i in range(12) :
        if i > 3 and k >= K :
            continue
        mx, my = x + dx[i], y + dy[i]
        if 0 <= mx < H and 0 <= my < W and board[mx][my] == 0 :
            if i > 3 :
                if dp[mx][my][k + 1] != 0 :
                    continue
                dp[mx][my][k + 1] = dp[x][y][k] + 1
                q.append([mx, my, k + 1])
            else :
                if dp[mx][my][k] != 0 :
                    continue
                dp[mx][my][k] = dp[x][y][k] + 1
                q.append([mx, my, k])

print(result)
