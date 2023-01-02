import sys

N, M = map(int, (sys.stdin.readline()).split())

board = [[0 for _ in range(N + 1)]]
for _ in range(N):
    board.append([0] + list(map(int, (sys.stdin.readline()).split())))

for i in range(1, N + 1):
    for j in range(1, N):
        board[i][j + 1] += board[i][j]

for j in range(1, N + 1):
    for i in range(1, N):
        board[i + 1][j] += board[i][j]

for _ in range(M):
    y1, x1, y2, x2 = map(int, (sys.stdin.readline()).split())
    print(board[y2][x2] - board[y1 - 1][x2] - board[y2][x1 - 1] + board[y1 - 1][x1 - 1])