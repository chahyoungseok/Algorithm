import sys
from collections import deque

N, M = map(int, sys.stdin.readline().strip().split())

dx, dy = [1, 0, -1, 0], [0, 1, 0, -1]
board = []
for _ in range(N):
    board.append(list(sys.stdin.readline().strip()))

max_value = 0
for i in range(N):
    for j in range(M):
        if board[i][j] == 'L':
            q = deque()
            q.append([i, j, 0])
            visited = [[True for _ in range(M)] for _ in range(N)]
            visited[i][j] = False

            while q:
                x, y, dist = q.popleft()

                if dist > max_value :
                    max_value = dist

                for k in range(4):
                    mx, my = dx[k] + x, dy[k] + y
                    if 0 <= mx < N and 0 <= my < M and board[mx][my] == 'L' and visited[mx][my]:
                        visited[mx][my] = False
                        q.append([mx, my, dist + 1])


print(max_value)