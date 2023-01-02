import sys
from collections import deque

N = int(sys.stdin.readline().strip())

board = []
for _ in range(N) :
    board.append(list(map(int, (sys.stdin.readline()).split())))

dx, dy = [1, -1, 0, 0], [0, 0, 1, -1]
max_count = 0

for h in range(1, 101) :
    visited = [[True for _ in range(N)] for _ in range(N)]
    count = 0
    for i in range(N) :
        for j in range(N) :
            if visited[i][j] and board[i][j] >= h:
                count += 1
                q = deque()
                q.append([i, j])

                while q :
                    x, y = q.popleft()

                    for k in range(4) :
                        mx, my = x + dx[k], y + dy[k]
                        if 0 <= mx < N and 0 <= my < N and visited[mx][my] and board[mx][my] >= h :
                            visited[mx][my] = False
                            if [mx, my] not in q :
                                q.append([mx, my])

    max_count = max(max_count, count)
print(max_count)