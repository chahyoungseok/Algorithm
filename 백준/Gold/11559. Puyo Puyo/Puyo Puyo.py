import sys
from collections import deque


def check() :
    global board
    visited = [[True for _ in range(6)] for _ in range(12)]
    trans_state = False

    for i in range(12) :
        for j in range(6) :
            if visited[i][j] and board[i][j] != ".":
                standard_color = board[i][j]
                q, visited[i][j], blocks = deque(), False, []
                q.append([i, j])

                while q :
                    x, y = q.popleft()
                    if [x, y] not in blocks :
                        blocks.append([x, y])

                    for k in range(4) :
                        mx, my = dx[k] + x, dy[k] + y
                        if 0 <= mx < 12 and 0 <= my < 6 and visited[mx][my] and standard_color == board[mx][my] :
                            visited[mx][my] = False
                            if [mx, my] not in q :
                                q.append([mx, my])

                if len(blocks) >= 4 :
                    trans_state = True
                    for x, y in blocks :
                        board[x][y] = "."
    return trans_state


def gravity() :
    global board

    for i in range(6) :
        keep = []
        for j in range(11, -1, -1) :
            if board[j][i] != "." :
                keep.append(board[j][i])
        keep_len = len(keep)
        for j in range(keep_len) :
            board[11 - j][i] = keep[j]
        for j in range(12 - keep_len) :
            board[j][i] = "."


board, total = [], 0
for _ in range(12) :
    board.append(list(sys.stdin.readline().strip()))

dx, dy = [1, -1, 0, 0], [0, 0, 1, -1]

while check() :
    total += 1
    gravity()

print(total)