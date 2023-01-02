import sys

N, M = map(int, (sys.stdin.readline()).split())
board = []
for _ in range(N) :
    board.append(list(map(int, (sys.stdin.readline()).split())))

sum_board = [[0 for _ in range(M + 1)]]
for i in range(N) :
    tmp = [0]
    for j in range(M) :
        tmp.append(tmp[-1] + board[i][j])
    sum_board.append(tmp)

for i in range(N) :
    for j in range(M + 1) :
        sum_board[i + 1][j] += sum_board[i][j]
        
K = int(sys.stdin.readline().strip())
for _ in range(K) :
    i, j, x, y = map(int, (sys.stdin.readline()).split())
    print(sum_board[x][y] + sum_board[i - 1][j - 1] - sum_board[x][j - 1] - sum_board[i - 1][y])
