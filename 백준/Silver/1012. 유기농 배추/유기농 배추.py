import sys
from collections import deque

T = int(input())

dx = [1, 0, -1, 0]
dy = [0, 1, 0, -1]

for _ in range(T) :
    M, N, K = map(int, (sys.stdin.readline()).split())
    board = [[0 for _ in range(M)] for _ in range(N)]
    for _ in range(K) :
        x, y = map(int, (sys.stdin.readline()).split())
        board[y][x] = 1

    total = 0
    for i in range(N) :
        for j in range(M) :
            if board[i][j] == 1 :
                q = deque()
                q.append([i, j])
                total += 1

                while q :
                    x, y = q.popleft()
                    board[x][y] = 0
                    for k in range(4) :
                        mx, my = x + dx[k], y + dy[k]
                        if mx >= 0 and mx < N and my >= 0 and my < M :
                            if board[mx][my] == 1 and not [mx, my] in q:
                                q.append([mx, my])
    print(total)