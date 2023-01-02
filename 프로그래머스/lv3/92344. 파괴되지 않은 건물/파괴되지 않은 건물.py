def solution(board, skill):
    board_len_1, board_len_2, count = len(board), len(board[0]), 0
    sum_board = [[0 for _ in range(board_len_2 + 1)] for _ in range(board_len_1 + 1)]

    for type, r1, c1, r2, c2, degree in skill :
        if type == 1 :
            degree = -degree

        sum_board[r1][c1] += degree
        sum_board[r1][c2 + 1] += -degree
        sum_board[r2 + 1][c1] += -degree
        sum_board[r2 + 1][c2 + 1] += degree

    for i in range(board_len_1) :
        for j in range(board_len_2) :
            sum_board[i][j + 1] += sum_board[i][j]

    for j in range(board_len_2) :
        for i in range(board_len_1) :
            sum_board[i + 1][j] += sum_board[i][j]
            
    for i in range(board_len_1) :
        for j in range(board_len_2) :
            if board[i][j] + sum_board[i][j] > 0 :
                count += 1

    return count