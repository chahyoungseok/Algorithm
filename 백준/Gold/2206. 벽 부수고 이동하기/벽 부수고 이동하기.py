import copy
from collections import deque

N, M = map(int, input().split())
board = []
for _ in range(N) :
    board.append(list(input()))

dx, dy = [1, 0, -1, 0], [0, 1, 0, -1]
q, visited = deque(), [[False for _ in range(M)] for _ in range(N)]
q.append([0 ,0, 1, 1])
visited[0][0], result = True, -1
b_visited = copy.deepcopy(visited)

while q :
    x, y, b, dist = q.popleft()

    if x == N - 1 and y == M - 1 :
        result = dist
        break

    for i in range(4) :
        mx, my = x + dx[i], y + dy[i]
        if mx >= 0 and mx < N and my >= 0 and my < M and not visited[mx][my] :
            if b == 0 and b_visited[mx][my] :
                continue
            if board[mx][my] == '0' :
                q.append([mx, my, b, dist + 1])
                b_visited[mx][my] = True
                if b == 1 :
                    visited[mx][my] = True
            elif board[mx][my] == '1' and b == 1 :
                q.append([mx, my, b - 1, dist + 1])
                b_visited[mx][my] = True
                visited[mx][my] = True

print(result)