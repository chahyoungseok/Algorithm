import sys
from collections import deque

board, empty_list = [], deque()
for i in range(9) :
    data = list(map(int, (sys.stdin.readline()).split()))
    for j in range(9):
        if data[j] == 0 :
            empty_list.append([i, j])
    board.append(data)


def find(f_board) :
    if not empty_list :
        for i in range(9) :
            for j in range(9) :
                print(f_board[i][j], end=" ")
            print()
        return "True"

    i, j = empty_list.popleft()
    candidate = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for s in range(9):
        if f_board[s][j] in candidate:
            candidate.remove(f_board[s][j])
        if f_board[i][s] in candidate:
            candidate.remove(f_board[i][s])

    if not candidate:
        empty_list.appendleft([i, j])
        return -1

    standard_i, standard_j = (i // 3) * 3, (j // 3) * 3
    for x in range(standard_i, standard_i + 3):
        for y in range(standard_j, standard_j + 3):
            if f_board[x][y] in candidate:
                candidate.remove(f_board[x][y])

    if not candidate:
        empty_list.appendleft([i, j])
        return -1

    for c in candidate:
        temp = f_board[i][j]
        f_board[i][j] = c
        if find(f_board) == "True":
            return "True"
        else:
            f_board[i][j] = temp

    empty_list.appendleft([i, j])
    return -1


find(board)
