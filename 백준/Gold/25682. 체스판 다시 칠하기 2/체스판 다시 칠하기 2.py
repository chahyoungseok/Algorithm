import sys

N, M, K = map(int, sys.stdin.readline().strip().split())

board, black_board, white_board = [], [[0 for _ in range(M + 1)] for _ in range(N + 1)], [[0 for _ in range(M + 1)] for _ in range(N + 1)],

for _ in range(N):
    board.append(list(sys.stdin.readline().strip()))

for i in range(N):
    for j in range(M):
        if (i + j) % 2 == 0:
            white_board[i + 1][j + 1] = int(board[i][j] == 'B')
            black_board[i + 1][j + 1] = int(board[i][j] == 'W')
        else:
            black_board[i + 1][j + 1] = int(board[i][j] == 'B')
            white_board[i + 1][j + 1] = int(board[i][j] == 'W')

        white_board[i + 1][j + 1] += (white_board[i + 1][j] + white_board[i][j + 1] - white_board[i][j])
        black_board[i + 1][j + 1] += (black_board[i + 1][j] + black_board[i][j + 1] - black_board[i][j])


min_value = sys.maxsize
for i in range(N - K + 1):
    for j in range(M - K + 1):

        white_value = white_board[i + K][j + K] + white_board[i][j] - white_board[i + K][j] - white_board[i][j + K]
        if min_value > white_value:
            min_value = white_value

        black_value = black_board[i + K][j + K] + black_board[i][j] - black_board[i + K][j] - black_board[i][j + K]
        if min_value > black_value:
            min_value = black_value

print(min_value)