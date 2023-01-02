import sys
from collections import deque

N, L, R = map(int, sys.stdin.readline().split())
board, result = [], 0

for _ in range(N) :
    board.append(list(map(int, (sys.stdin.readline()).split())))

dx, dy = [1, -1, 0, 0], [0, 0, 1, -1]

while True :
    visited = [[True for _ in range(N)] for _ in range(N)]
    state, q = True, deque()
    for i in range(N) :
        for j in range(N) :
            if visited[i][j] :
                total_p, total_c = 0, []
                q.append([i, j])
                visited[i][j] = False

                while q :
                    x, y = q.popleft()
                    if [x, y] not in total_c :
                        total_p += board[x][y]
                        total_c.append([x, y])

                    for k in range(4) :
                        mx, my = x + dx[k], y + dy[k]
                        if 0 <= mx < N and 0 <= my < N and visited[mx][my] :
                            if L <= abs(board[x][y] - board[mx][my]) <= R :
                                q.append([mx, my])
                                visited[mx][my] = False

                t_c = len(total_c)
                if t_c > 1 :
                    state = False
                    average_p = int(total_p / t_c)
                    for x, y in total_c:
                        board[x][y] = average_p

    if state :
        break
    result += 1


print(result)
