import copy, sys

R, C, M = map(int, (sys.stdin.readline()).split())
board = [[[] for _ in range(C + 1)] for _ in range(R + 1)]
total = 0

dx, dy = [-1, 1, 0, 0], [0, 0, 1, -1]

for _ in range(M) :
    r, c, s, d, z = map(int, (sys.stdin.readline()).split())
    board[r][c].append([s, d, z])
    
for i in range(1, C + 1) :
    for j in range(1, R + 1) :
        if board[j][i]:
            total += board[j][i].pop()[2]
            break

    temp_board = [[[] for _ in range(C + 1)] for _ in range(R + 1)]
    for a in range(1, R + 1) :
        for b in range(1, C + 1) :
            if board[a][b] :
                s, d, z = board[a][b].pop()
                mx, my = a + dx[d - 1] * s, b + dy[d - 1] * s
                if d > 2 :
                    if my < 0:
                        my = -(-my % ((C - 1) * 2))
                    else:
                        my %= (C - 1) * 2
                    state = 0
                    if my <= 0 :
                        state = 1
                        my = -(my - 2)
                    if my > C:
                        state += (my - 2) // (C - 1)
                        if state % 2 == 1 :
                            my = C - ((my - 2) % (C - 1) + 1)
                        else :
                            if d == 3 :
                                my = (my - 2) % (C - 1) + 2
                            else :
                                my = C - ((my - 2) % (C - 1) + 1)

                    if state % 2 == 0 :
                        temp_board[mx][my].append([s, d, z])
                    else :
                        if d == 3 :
                            d = 4
                        else :
                            d = 3
                        temp_board[mx][my].append([s, d, z])
                else :
                    if mx < 0:
                        mx = -(-mx % ((R - 1) * 2))
                    else:
                        mx %= (R - 1) * 2
                    state = 0
                    if mx <= 0 :
                        state = 1
                        mx = -(mx - 2)
                    if mx > R:
                        state += (mx - 2) // (R - 1)
                        if state % 2 == 1 :
                            mx = R - ((mx - 2) % (R - 1) + 1)
                        else :
                            if d == 2 :
                                mx = (mx - 2) % (R - 1) + 2
                            else :
                                mx = R - ((mx - 2) % (R - 1) + 1)

                    if state % 2 == 0:
                        temp_board[mx][my].append([s, d, z])
                    else :
                        if d == 1 :
                            d = 2
                        else :
                            d = 1
                        temp_board[mx][my].append([s, d, z])
                if len(temp_board[mx][my]) >= 2 :
                    temp_board[mx][my] = sorted(temp_board[mx][my], key=lambda x : -x[2])
                    temp_board[mx][my] = [temp_board[mx][my][0]]

    board = copy.deepcopy(temp_board)
print(total)
