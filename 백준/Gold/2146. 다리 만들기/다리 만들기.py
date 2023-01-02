import sys
from collections import deque

N = int(sys.stdin.readline().strip())
board = []
for _ in range(N) :
    board.append(list(map(int, (sys.stdin.readline()).split())))

dx, dy = [1, -1, 0, 0], [0, 0, 1, -1]

res = 1
tmp_board = [[0 for _ in range(N)] for _ in range(N)]
for i in range(N) :
    for j in range(N) :
        if board[i][j] == 1 and tmp_board[i][j] == 0 :
            q = deque()
            q.append([i, j])
            tmp_board[i][j] = res

            while q :
                x, y = q.popleft()

                for k in range(4) :
                    mx, my = x + dx[k], y + dy[k]
                    if 0 <= mx < N and 0 <= my < N and tmp_board[mx][my] == 0 and board[mx][my] == 1:
                        if [mx, my] not in q :
                            q.append([mx, my])
                        tmp_board[mx][my] = res
            res += 1

min_dist = int(1e9)
for i in range(1, res) :
    for j in range(N) :
        for k in range(N) :
            if tmp_board[j][k] == i :
                q = deque()
                q.append([j, k, 0])

                visited = [[True for _ in range(N)] for _ in range(N)]
                visited[j][k] = False

                state = True
                while q and state:
                    x, y, dist = q.popleft()

                    for d in range(4) :
                        mx, my = x + dx[d], y + dy[d]
                        if 0 <= mx < N and 0 <= my < N and tmp_board[mx][my] != i and visited[mx][my]:
                            if board[mx][my] != 0 :
                                min_dist = min(min_dist, dist)
                                state = False
                                break
                            q.append([mx, my, dist + 1])
                            visited[mx][my] = False

print(min_dist)