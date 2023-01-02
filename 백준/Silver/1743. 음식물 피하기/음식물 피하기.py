import sys
from collections import deque

N, M, K = map(int, (sys.stdin.readline()).split())
board = [[0 for _ in range(M)] for _ in range(N)]
for i in range(K) :
    r, c = map(int, (sys.stdin.readline()).split())
    board[r - 1][c - 1] = 1

dx, dy = [1, -1, 0, 0], [0, 0, 1, -1]
max_total, visited = 0, [[True for _ in range(M)] for _ in range(N)]
q = deque()
for i in range(N) :
    for j in range(M) :
        if visited[i][j] and board[i][j] == 1:
            q.append([i, j])
            visited[i][j], total = False, 0
            while q :
                x, y = q.popleft()
                total += 1

                for k in range(4) :
                    mx, my = x + dx[k], y + dy[k]
                    if 0 <= mx < N and 0 <= my < M and visited[mx][my] and board[mx][my] == 1:
                        visited[mx][my] = False
                        q.append([mx, my])
            max_total = max(max_total, total)
print(max_total)