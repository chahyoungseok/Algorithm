import sys

N = int(sys.stdin.readline())
total, board = 0, []
for _ in range(N) :
    board.append(list(map(int, (sys.stdin.readline()).split())))

if board[N - 1][N - 1] == 1 or (board[N - 2][N - 1] == 1 and board[N - 1][N - 2] == 1):
    print(0)
else :
    current_ap = 0
    dx, dy = [[0, 1], [1, 1], [0, 1, 1]], [[1, 1], [0, 1], [1, 0, 1]]


    def dfs(x, y, ap) :
        if x == N - 1 and y == N - 1 :
            global total
            total += 1
            return

        if (ap == 0 and y == N - 1) or (ap == 1 and x == N - 1):
            return

        if ap == 2 :
            for i in range(3) :
                mx, my = x + dx[ap][i], y + dy[ap][i]
                if mx < N and my < N and board[mx][my] == 0:
                    if i == 2:
                        if not (board[mx - 1][my] == 0 and board[mx][my - 1] == 0):
                            continue
                    dfs(mx, my, i)
        else :
            for i in range(2) :
                mx, my = x + dx[ap][i], y + dy[ap][i]
                if mx < N and my < N and board[mx][my] == 0:
                    if i == 1:
                        if not (board[mx - 1][my] == 0 and board[mx][my - 1] == 0):
                            continue
                    if i == 0 :
                        dfs(mx, my, ap)
                    else :
                        dfs(mx, my, 2)


    dfs(0, 1, 0)
    print(total)