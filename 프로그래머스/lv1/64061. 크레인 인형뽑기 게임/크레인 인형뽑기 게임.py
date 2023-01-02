def solution(board, moves):
    board_len = len(board)
    basket, result = [], 0

    for move in moves:
        for i in range(board_len):
            if board[i][move - 1] != 0:
                if basket and basket[len(basket) - 1] == board[i][move - 1]:
                    basket.pop()
                    result += 2
                else:
                    basket.append(board[i][move - 1])
                board[i][move - 1] = 0
                break

    return result