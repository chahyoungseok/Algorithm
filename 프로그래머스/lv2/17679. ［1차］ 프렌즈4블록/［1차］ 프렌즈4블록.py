import copy


def solution(m, n, board):
    confirm_shape = [[0, 0], [0, 1], [1, 0], [1, 1]]

    for i in range(m) :
        board[i] = list(board[i])

    def check(board):
        temp, trans_state = copy.deepcopy(board), False

        for x in range(m - 1):
            for y in range(n - 1):
                check_len, standard = 3, board[x][y]
                if standard == "" :
                    continue
                for k in range(1, 4):
                    a, b = confirm_shape[k]

                    if board[a + x][b + y] == standard:
                        check_len -= 1

                if check_len == 0:
                    trans_state = True
                    temp[x][y], temp[x + 1][y], temp[x][y + 1], temp[x + 1][y + 1] = "", "", "", ""

        return trans_state, temp

    def gravity(board):
        for x in range(n):
            word = ""
            for y in range(m - 1, -1, -1):
                word += board[y][x]

            word_len = len(word)
            for y in range(word_len):
                board[m - y - 1][x] = word[y]

            for y in range(m - word_len):
                board[y][x] = ""
        return board


    trans, board = check(board)
    while trans:
        board = gravity(board)
        trans, board = check(board)

    answer = 0
    for i in range(m) :
        for j in range(n) :
            if board[i][j] == "" :
                answer += 1

    return answer