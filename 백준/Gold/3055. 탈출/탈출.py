import copy, sys
from collections import deque

R, C = map(int, (sys.stdin.readline()).split())
board, waves = [], []
start = [0, 0]
for i in range(R) :
    data = list(sys.stdin.readline().strip())
    board.append(data)
    for j in range(C) :
        if data[j] == 'S' :
            start = [i, j]
        elif data[j] == '*' :
            waves.append([i, j])

dx, dy = [1, -1, 0, 0], [0, 0, 1, -1]

q = deque()
q.append([start[0], start[1], 0])
visited = [[True for _ in range(C)] for _ in range(R)]
visited[start[0]][start[1]] = False

tmp_waves, state, current_dist = [], True, 0
while q and state:
    x, y, dist = q.popleft()

    if dist > current_dist :
        current_dist = dist
        waves = copy.deepcopy(tmp_waves)

    for wx, wy in waves :
        for i in range(4) :
            mx, my = wx + dx[i], wy + dy[i]
            if 0 <= mx < R and 0 <= my < C and board[mx][my] == '.':
                tmp_waves.append([mx, my])
                board[mx][my] = '*'

    for i in range(4) :
        mx, my = x + dx[i], y + dy[i]
        if 0 <= mx < R and 0 <= my < C and visited[mx][my] :
            if board[mx][my] == 'D' :
                print(dist + 1)
                state = False
                break
            elif board[mx][my] == '*' or board[mx][my] == 'X':
                continue
            q.append([mx, my, dist + 1])
            visited[mx][my] = False
if state :
    print("KAKTUS")
