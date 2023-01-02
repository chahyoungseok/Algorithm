import copy
import sys
from itertools import combinations
from collections import deque

N, M = map(int, input().split())

board = []
for _ in range(N) :
    board.append(list(map(int, (sys.stdin.readline()).split())))

wall_total, wall_p, virus = 0, [], []
for i in range(N) :
    for j in range(M) :
        if board[i][j] == 2 :
            virus.append([i, j])
        elif board[i][j] == 0 :
            wall_p.append([i, j])
        else :
            wall_total += 1

safe_zoon, dx, dy = [], [1, -1, 0, 0], [0, 0, 1, -1]
for one, two, three in combinations(wall_p, 3) :
    new_board = copy.deepcopy(board)
    new_board[one[0]][one[1]] = 1
    new_board[two[0]][two[1]] = 1
    new_board[three[0]][three[1]] = 1

    q, non_safe_total = deque(virus), 0

    while q:
        x, y = q.popleft()
        non_safe_total += 1

        for j in range(4):
            mx, my = x + dx[j], y + dy[j]
            if mx >= 0 and mx < N and my >= 0 and my < M:
                if new_board[mx][my] == 0 :
                    new_board[mx][my] = 2
                    q.append([mx, my])

    safe_zoon.append(N * M - non_safe_total - wall_total - 3)
print(max(safe_zoon))