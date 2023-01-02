import sys
from collections import deque

M, N = map(int, (sys.stdin.readline()).split())
board = []
for _ in range(N) :
    board.append(list(sys.stdin.readline().strip()))

dx, dy = [1, -1, 0, 0], [0, 0, 1, -1]
visited = [[True for _ in range(M)] for _ in range(N)]
a_total, b_total = 0, 0

for i in range(N) :
    for j in range(M) :
        if visited[i][j] :
            visited[i][j], total = False, 0
            q = deque()
            q.append([i, j])
            while q :
                x, y = q.popleft()
                total += 1

                for k in range(4) :
                    mx, my = x + dx[k], y + dy[k]
                    if 0 <= mx < N and 0 <= my < M and visited[mx][my] and board[i][j] == board[mx][my]:
                        visited[mx][my] = False
                        q.append([mx, my])

            if board[i][j] == 'B' :
                b_total += total ** 2
            else :
                a_total += total ** 2

print(str(a_total) + " " + str(b_total))