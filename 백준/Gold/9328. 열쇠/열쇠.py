import sys
from collections import deque

T = int(sys.stdin.readline().strip())
dx, dy = [1, -1, 0, 0], [0, 0, 1, -1]

for _ in range(T) :
    h, w = map(int, (sys.stdin.readline()).split())

    board, count, keys = [], 0, []
    for _ in range(h) :
        board.append(list(sys.stdin.readline().strip()))

    keys_str = sys.stdin.readline().strip()
    if keys_str == "0" :
        keys = []
    else :
        for i in list(keys_str) :
            keys.append(ord(i))

    q, doors = deque(), deque()
    visited = [[True for _ in range(w)] for _ in range(h)]
    for i in range(h) :
        if i == 0 or i == h - 1:
            for j in range(w) :
                if board[i][j] != "*" :
                    if 65 <= ord(board[i][j]) <= 90:
                        if ord(board[i][j]) + 32 not in keys :
                            doors.append([i, j, ord(board[i][j]) + 32])
                            continue
                    elif 97 <= ord(board[i][j]) <= 122 :
                        target = ord(board[i][j])
                        keys.append(target)
                        for d in range(len(doors) - 1, -1, -1) :
                            if doors[d][2] == target :
                                if [doors[d][0], doors[d][1]] not in q :
                                    q.append([doors[d][0], doors[d][1]])
                                    visited[doors[d][0]][doors[d][1]] = False
                                    doors.remove(doors[d])
                    elif board[i][j] == "$" :
                        count += 1
                    q.append([i, j])
                    visited[i][j] = False

        else :
            w_list = [0, w - 1]
            for j in w_list :
                if board[i][j] != "*":
                    if 65 <= ord(board[i][j]) <= 90:
                        if ord(board[i][j]) + 32 not in keys:
                            doors.append([i, j, ord(board[i][j]) + 32])
                            continue
                    elif 97 <= ord(board[i][j]) <= 122:
                        target = ord(board[i][j])
                        keys.append(target)
                        for d in range(len(doors) - 1, -1, -1):
                            if doors[d][2] == target:
                                if [doors[d][0], doors[d][1]] not in q:
                                    q.append([doors[d][0], doors[d][1]])
                                    visited[doors[d][0]][doors[d][1]] = False
                                    doors.remove(doors[d])
                    elif board[i][j] == "$":
                        count += 1
                    q.append([i, j])
                    visited[i][j] = False

    while q:
        x, y = q.popleft()

        for k in range(4):
            mx, my = x + dx[k], y + dy[k]
            if 0 <= mx < h and 0 <= my < w and board[mx][my] != "*" and visited[mx][my]:
                if 65 <= ord(board[mx][my]) <= 90:
                    if ord(board[mx][my]) + 32 not in keys:
                        doors.append([mx, my, ord(board[mx][my]) + 32])
                        continue
                elif 97 <= ord(board[mx][my]) <= 122:
                    target = ord(board[mx][my])
                    keys.append(target)
                    for d in range(len(doors) - 1, -1, -1):
                        if doors[d][2] == target:
                            if [doors[d][0], doors[d][1]] not in q:
                                q.append([doors[d][0], doors[d][1]])
                                visited[doors[d][0]][doors[d][1]] = False
                                doors.remove(doors[d])
                elif board[mx][my] == "$":
                    count += 1
                q.append([mx, my])
                visited[mx][my] = False

    print(count)
