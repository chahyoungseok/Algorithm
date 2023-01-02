import sys
from collections import deque

N, M = map(int, (sys.stdin.readline()).split())
board, count, state = [], 0, False
for _ in range(N) :
    board.append(list(map(int, (sys.stdin.readline()).split())))

dx, dy = [1, -1, 0, 0], [0, 0, 1, -1]

for k in range(N):
    if sum(board[k]) != 0:
        state = True

if state :
    while True :
        q, state = deque(), True
        q.append([0, 0])
        visited, temp_board = [[True for _ in range(M)] for _ in range(N)], [[0 for _ in range(M)] for _ in range(N)]
        visited[0][0] = False
        while q:
            x, y = q.popleft()

            for i in range(4):
                mx, my = x + dx[i], y + dy[i]
                if 0 <= mx < N and 0 <= my < M and visited[mx][my] and board[mx][my] == 0:
                    visited[mx][my] = False
                    q.append([mx, my])
                    for j in range(4):
                        tx, ty = mx + dx[j], my + dy[j]
                        if 0 <= tx < N and 0 <= ty < M and board[tx][ty] == 1:
                            temp_board[tx][ty] += 1

        for i in range(N):
            for j in range(M):
                if temp_board[i][j] >= 2:
                    board[i][j] = 0
                elif board[i][j] != 0:
                    state = False

        count += 1
        if state :
            break

print(count)