import copy


def move(state, real_board, dist) :
    global max_value
    board = copy.deepcopy(real_board)

    if state == "up" :
        for i in range(N) :
            for j in range(N - 1) :
                if board[j][i] == 0 :
                    for k in range(j + 1, N) :
                        if board[k][i] != 0 :
                            board[j][i] = board[k][i]
                            board[k][i] = 0
                            break
                    if board[j][i] == 0 :
                        break
            for j in range(N - 1) :
                if board[j][i] == 0 or board[j + 1][i] == 0 :
                    continue
                if board[j][i] == board[j + 1][i] :
                    board[j][i] *= 2
                    board[j + 1][i] = 0
            for j in range(N - 1) :
                if board[j][i] == 0 :
                    for k in range(j + 1, N) :
                        if board[k][i] != 0 :
                            board[j][i] = board[k][i]
                            board[k][i] = 0
                            break
                    if board[j][i] == 0 :
                        break
    elif state == "down" :
        for i in range(N - 1, -1, -1) :
            for j in range(N - 1, 0, -1) :
                if board[j][i] == 0 :
                    for k in range(j - 1, -1, -1) :
                        if board[k][i] != 0 :
                            board[j][i] = board[k][i]
                            board[k][i] = 0
                            break
                    if board[j][i] == 0 :
                        break
            for j in range(N - 1, 0, -1) :
                if board[j][i] == 0 or board[j - 1][i] == 0 :
                    continue
                if board[j][i] == board[j - 1][i] :
                    board[j][i] *= 2
                    board[j - 1][i] = 0
            for j in range(N - 1, 0, -1) :
                if board[j][i] == 0 :
                    for k in range(j - 1, -1, -1) :
                        if board[k][i] != 0 :
                            board[j][i] = board[k][i]
                            board[k][i] = 0
                            break
                    if board[j][i] == 0 :
                        break
    elif state == "left" :
        for j in range(N) :
            for i in range(N - 1) :
                if board[j][i] == 0 :
                    for k in range(i + 1, N) :
                        if board[j][k] != 0 :
                            board[j][i] = board[j][k]
                            board[j][k] = 0
                            break
                    if board[j][i] == 0 :
                        break
            for i in range(N - 1) :
                if board[j][i] == 0 or board[j][i + 1] == 0 :
                    continue
                if board[j][i] == board[j][i + 1] :
                    board[j][i] *= 2
                    board[j][i + 1] = 0
            for i in range(N - 1) :
                if board[j][i] == 0 :
                    for k in range(i + 1, N) :
                        if board[j][k] != 0 :
                            board[j][i] = board[j][k]
                            board[j][k] = 0
                            break
                    if board[j][i] == 0 :
                        break
    elif state == "right" :
        for j in range(N - 1, -1, -1):
            for i in range(N - 1, 0, -1) :
                if board[j][i] == 0 :
                    for k in range(i - 1, -1, -1) :
                        if board[j][k] != 0 :
                            board[j][i] = board[j][k]
                            board[j][k] = 0
                            break
                    if board[j][i] == 0 :
                        break
            for i in range(N - 1, 0, -1):
                if board[j][i] == 0 or board[j][i - 1] == 0:
                    continue
                if board[j][i] == board[j][i - 1]:
                    board[j][i] *= 2
                    board[j][i - 1] = 0
            for i in range(N - 1, 0, -1) :
                if board[j][i] == 0 :
                    for k in range(i - 1, -1, -1) :
                        if board[j][k] != 0 :
                            board[j][i] = board[j][k]
                            board[j][k] = 0
                            break
                    if board[j][i] == 0 :
                        break

    if dist == 4 :
        for i in range(N) :
            max_value = max(max_value, max(board[i]))
        return

    move("up", board, dist + 1)
    move("down", board, dist + 1)
    move("left", board, dist + 1)
    move("right", board, dist + 1)


N = int(input())

o_board, max_value = [], 0
for _ in range(N) :
    o_board.append(list(map(int, input().split())))

move("up", o_board, 0)
move("down", o_board, 0)
move("left", o_board, 0)
move("right", o_board, 0)

print(max_value)
