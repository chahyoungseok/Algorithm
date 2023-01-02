import sys

N, M = map(int, (sys.stdin.readline()).split())
current_x, current_y, direction = map(int, (sys.stdin.readline()).split())

board = []
dx, dy = [-1, 0, 1, 0], [0, 1, 0, -1]
for _ in range(N) :
    board.append(list(map(int, (sys.stdin.readline()).split())))

result, state_index, dist = int(1e9), 0, 1
board[current_x][current_y] = 2

while True :
    if state_index >= 4 :
        mx, my = current_x - dx[direction], current_y - dy[direction]
        if board[mx][my] == 1 :
            result = dist
            break
        else :
            current_x, current_y, state_index = mx, my, 0
            continue

    direction = (direction - 1 + 4) % 4
    mx, my = current_x + dx[direction], current_y + dy[direction]
    if 0 <= mx < N and 0 <= my < M and board[mx][my] == 0:
        board[mx][my] = 2
        current_x, current_y, state_index, dist = mx, my, 0, dist + 1
        continue
    else:
        state_index += 1

print(result)