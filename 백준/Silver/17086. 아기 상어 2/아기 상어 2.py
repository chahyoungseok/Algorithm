import sys
from collections import deque

N, M = map(int, (sys.stdin.readline()).split())
board = []
for _ in range(N) :
    board.append(list(map(int, (sys.stdin.readline()).split())))

dx, dy = [1, -1, 0, 0, 1, 1, -1, -1], [0, 0, -1, 1, 1, -1, 1, -1]
max_total = 0


def bfs(i_b, j_b) :
    q = deque()
    q.append([i_b, j_b, 0])
    visited = [[True for _ in range(M)] for _ in range(N)]
    visited[i_b][j_b] = False
    while q:
        x, y, dist = q.popleft()

        for k in range(8):
            mx, my = x + dx[k], y + dy[k]
            if 0 <= mx < N and 0 <= my < M and visited[mx][my]:
                if board[mx][my] == 1:
                    return dist + 1
                visited[mx][my] = False
                q.append([mx, my, dist + 1])


for i in range(N) :
    for j in range(M) :
        if board[i][j] == 0 :
            max_total = max(max_total, bfs(i, j))

print(max_total)