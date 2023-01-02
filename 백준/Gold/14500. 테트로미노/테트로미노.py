import sys
from itertools import combinations

max_total = 0
sp_case = [[1, 0], [-1, 0], [0, 1], [0, -1]]


def dfs(x, y, visited, result, dist) :
    global max_total, max_board

    if max_total >= result + max_board * (4 - dist) :
        return

    for k in range(4) :
        mx, my = x + dx[k], y + dy[k]
        if mx >= 0 and mx < N and my >= 0 and my < M and visited[mx][my] :
            if dist == 3 :
                max_total = max(max_total, result + board[mx][my])
                continue
            visited[mx][my] = False
            dfs(mx, my, visited, result + board[mx][my], dist + 1)
            visited[mx][my] = True


N, M = map(int, input().split())
board, max_data = [], []
dx, dy = [1, -1, 0, 0], [0, 0, 1, -1]

for _ in range(N) :
    data = list(map(int, (sys.stdin.readline()).split()))
    max_data.append(max(data))
    board.append(data)

visited, max_board = [[True for _ in range(M)] for _ in range(N)], max(max_data)

for i in range(N) :
    for j in range(M) :
        visited[i][j], result = False, board[i][j]
        dfs(i,j, visited, result, 1)
        visited[i][j] = True

        for case in combinations(sp_case, 3) :
            sp_sum = board[i][j]
            for s_x, s_y in case :
                mx, my = i + s_x, j + s_y
                if mx >= 0 and mx < N and my >= 0 and my < M :
                    sp_sum += board[mx][my]
                else :
                    break
            max_total = max(max_total, sp_sum)

print(max_total)