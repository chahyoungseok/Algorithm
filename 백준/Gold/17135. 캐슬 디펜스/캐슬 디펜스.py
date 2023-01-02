import sys, copy
from itertools import combinations
from collections import deque


def attack(attacker, copy_board) :
    remove_list, set_remove = [], set()
    for y in attacker :
        visited = [[True for _ in range(M)] for _ in range(N + 1)]
        visited[N][y] = False

        state = True

        q = deque()
        q.append([N, y, 0])

        while q and state:
            c_x, c_y, dist = q.popleft()

            if dist + 1 > D :
                break

            for k in range(3) :
                mx, my = c_x + dx[k], c_y + dy[k]
                if 0 <= mx < N and 0 <= my < M and visited[mx][my] :
                    if copy_board[mx][my] == 0 :
                        q.append([mx, my, dist + 1])
                        visited[mx][my] = False
                    else :
                        remove_list.append([mx, my])
                        state = False
                        break

    for x, y in remove_list :
        set_remove.add((x, y))
        copy_board[x][y] = 0

    global count
    count += len(set_remove)

    for i in range(N) :
        if sum(copy_board[i]) > 0 :
            return False
    return True


def move(copy_board) :
    for i in range(N - 1, -1, -1) :
        for j in range(M) :
            if i == N - 1 :
                copy_board[i][j] = 0
            else :
                if copy_board[i][j] == 1 :
                    copy_board[i + 1][j] = 1
                    copy_board[i][j] = 0

    for i in range(N) :
        if sum(copy_board[i]) > 0 :
            return False
    return True


N, M, D = map(int, (sys.stdin.readline()).split())
board, max_count = [], 0
for _ in range(N) :
    board.append(list(map(int, (sys.stdin.readline()).split())))
board.append([0 for _ in range(M)])

dx, dy = [0, -1, 0], [-1, 0, 1]

for comb in combinations(list(range(M)), 3) :
    count, copy_board = 0, copy.deepcopy(board)
    while True :
        if attack(list(comb), copy_board) :
            break
        if move(copy_board) :
            break
    max_count = max(max_count, count)

print(max_count)