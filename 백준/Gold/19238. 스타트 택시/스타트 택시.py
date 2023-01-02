import sys
from collections import deque

N, M, f = map(int, (sys.stdin.readline()).split())
board, count, info, state = [], 0, [[[] for _ in range(N)] for _ in range(N)], True

for _ in range(N) :
    board.append(list(map(int, (sys.stdin.readline()).split())))

x, y = map(int, (sys.stdin.readline()).split())
x, y = x - 1, y - 1

dx, dy = [1, -1, 0, 0], [0, 0, 1, -1]

for _ in range(M) :
    sx, sy, ex, ey = map(int, (sys.stdin.readline()).split())
    info[sx - 1][sy - 1].append([ex - 1, ey - 1])

while count != M :
    q = deque()
    q.append([x, y, 0])
    visited = [[True for _ in range(N)] for _ in range(N)]
    result = []

    while q :
        x, y, dist = q.popleft()

        if info[x][y] :
            if not result :
                result = [dist, x, y]
            elif result[0] == dist :
                if result[1] > x :
                    result = [dist, x, y]
                elif result[1] == x :
                    if result[2] > y :
                        result = [dist, x, y]
            else :
                break

        for i in range(4) :
            mx, my = x + dx[i], y + dy[i]
            if 0 <= mx < N and 0 <= my < N and visited[mx][my] and board[mx][my] != 1:
                visited[mx][my] = False
                if [mx, my] not in q :
                    q.append([mx, my, dist + 1])

    if result and f >= result[0] :
        f -= result[0]
        q = deque()
        q.append([result[1], result[2], 0])
        visited = [[True for _ in range(N)] for _ in range(N)]
        tx, ty = info[result[1]][result[2]][0]
        min_dist = int(1e9)

        while q:
            x, y, dist = q.popleft()

            if x == tx and y == ty :
                min_dist = dist
                break

            for i in range(4):
                mx, my = x + dx[i], y + dy[i]
                if 0 <= mx < N and 0 <= my < N and visited[mx][my] and board[mx][my] != 1:
                    visited[mx][my] = False
                    if [mx, my] not in q:
                        q.append([mx, my, dist + 1])

        if f >= min_dist :
            f += min_dist
            count += 1
            info[result[1]][result[2]] = []
        else :
            state = False
            break
    else :
        state = False
        break

if state :
    print(f)
else :
    print(-1)