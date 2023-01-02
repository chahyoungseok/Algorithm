import sys

max_distance = 0


def dfs(x, y, visited, dist):
    global max_distance

    max_distance = max(max_distance, dist)

    for i in range(4):
        mx, my = x + dx[i], y + dy[i]
        if mx >= 0 and mx < R and my >= 0 and my < C and board[mx][my] not in visited:
            visited.add(board[mx][my])
            dfs(mx, my, visited, dist + 1)
            visited.remove(board[mx][my])


R, C = map(int, input().split())

board = []
for _ in range(R):
    board.append(list(sys.stdin.readline().rstrip()))

visited = set()
visited.add(board[0][0])
dx, dy = [-1, 1, 0, 0], [0, 0, -1, 1]
dfs(0, 0, visited, 1)
print(max_distance)