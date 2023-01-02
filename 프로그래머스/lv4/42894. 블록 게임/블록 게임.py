import copy


def solution(board):
    answer, N = 0, len(board)
    need_block, r_list = {}, ["n"]

    for x in range(N):
        for y in range(N):
            if board[x][y] != 0:
                if board[x][y] in need_block.keys():
                    need_block[board[x][y]].append([x, y])
                else:
                    need_block[board[x][y]] = [[x, y]]

    for key in need_block.keys():
        target = need_block[key]
        need_block[key].append(
            [min(target, key=lambda x: x[0])[0], max(target, key=lambda x: x[0])[0], min(target, key=lambda x: x[1])[1],
             max(target, key=lambda x: x[1])[1]])

    while r_list:

        def test(N, board_, need_block):
            tmp_board = copy.deepcopy(board_)
            remove_index = []

            for _ in range(2):
                for j in range(N):
                    for i in range(N):
                        if tmp_board[i][j] == 0 :
                            continue
                        if i != 0:
                            tmp_board[i - 1][j] = -1
                        break
            for key in need_block.keys():
                x1, x2, y1, y2 = need_block[key][4]
                state = True
                for i in range(x1, x2 + 1):
                    for j in range(y1, y2 + 1):
                        if not (tmp_board[i][j] == -1 or tmp_board[i][j] == key):
                            state = False
                            break
                    if not state:
                        break
                if state:
                    remove_index.append(key)
            return remove_index

        r_list = test(N, board, need_block)
        if r_list :
            answer += len(r_list)
            for r in r_list:
                del need_block[r]
            for i in range(N):
                for j in range(N):
                    if board[i][j] in r_list:
                        board[i][j] = 0

    return answer