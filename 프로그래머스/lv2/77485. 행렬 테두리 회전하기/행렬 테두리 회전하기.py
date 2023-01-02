def cycle(board, query) :
    x1, y1, x2, y2 = query
    min_sel = int(1e9)

    temp = board[x1 - 1][y1 - 1]
    for i in range(x1 - 1, x2 - 1):
        board[i][y1 - 1] = board[i + 1][y1 - 1]
        if min_sel > board[i][y1 - 1]:
            min_sel = board[i][y1 - 1]

    for i in range(y1 - 1, y2 - 1):
        board[x2 - 1][i] = board[x2 - 1][i + 1]
        if min_sel > board[x2 - 1][i]:
            min_sel = board[x2 - 1][i]

    for i in range(x2 - 1, x1 - 1, -1):
        board[i][y2 - 1] = board[i - 1][y2 - 1]
        if min_sel > board[i][y2 - 1]:
            min_sel = board[i][y2 - 1]

    for i in range(y2 - 1, y1 - 1, -1):
        board[x1 - 1][i] = board[x1 - 1][i - 1]
        if min_sel > board[x1 - 1][i]:
            min_sel = board[x1 - 1][i]

    board[x1 - 1][y1] = temp

    return min(min_sel, temp)

        
def solution(rows, columns, queries):
    answer = []
    board = []
    
    for i in range(rows) :
        pre = []
        for j in range(columns) :
            pre.append((i * columns + j) + 1)
        board.append(pre)
        
    for query in queries :
        answer.append(cycle(board, query))
        
    
    return answer