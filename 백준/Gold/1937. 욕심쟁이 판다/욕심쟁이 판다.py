import sys
sys.setrecursionlimit(int(1e6))

n = int(sys.stdin.readline().strip())
board, max_distance = [], 0
dx, dy = [0, 0, 1, -1], [1, -1, 0, 0]
dp = [[1 for _ in range(n)] for _ in range(n)]

for _ in range(n) :
    board.append(list(map(int, (sys.stdin.readline()).split())))


def dfs(x, y):
    if dp[x][y] != 1 :
        return dp[x][y]

    for k in range(4) :
        mx, my = x + dx[k], y + dy[k]
        if 0 <= mx < n and 0 <= my < n and board[mx][my] > board[x][y] :
            dp[x][y] = max(dp[x][y], dfs(mx, my) + 1)

    return dp[x][y]


for i in range(n) :
    for j in range(n) :
        dp[i][j] = max(dp[i][j], dfs(i, j))

for i in range(n) :
    max_distance = max(max_distance, max(dp[i]))
    
print(max_distance)