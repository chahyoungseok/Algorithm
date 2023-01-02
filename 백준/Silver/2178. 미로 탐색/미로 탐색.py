import sys
from collections import deque

input = sys.stdin.readline
N, M = map(int, input().split())

board = []
for _ in range(N) :
    board.append(list(input()))

dx = [1, 0, -1, 0]
dy = [0, 1, 0, -1]
visited = [[False for _ in range(M)] for _ in range(N)]
visited[0][0] = True
q = deque()
q.append([0, 0, 1])

while q :
    x, y, dist = q.popleft()

    if x == N - 1 and y == M - 1 :
        print(dist)
        break

    for i in range(4) :
        mx, my = x + dx[i], y + dy[i]
        if mx >=0 and mx < N and my >=0 and my < M :
            if board[mx][my] == '1' and not visited[mx][my] :
                visited[mx][my] = True
                q.append([mx, my, dist + 1])

