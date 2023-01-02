import sys

R, C = map(int, (sys.stdin.readline()).split())
board, count, dx = [], 0, [-1, 0, 1]
for _ in range(R) :
    board.append(list(sys.stdin.readline().strip()))


def dfs(x, y, board) :
    if y == C - 1 :
        return True

    board[x][y] = 'x'
    for k in range(3) :
        mx, my = x + dx[k], y + 1
        if 0 <= mx < R and my < C and board[mx][my] == ".":
            if dfs(mx, my, board) :
                return True
    return False


for i in range(R) :
    if dfs(i, 0, board) :
        count += 1
print(count)