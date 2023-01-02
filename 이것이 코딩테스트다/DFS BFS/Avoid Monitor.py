from itertools import combinations
import copy

N = int(input())
corridor, blank_list, student_list, teacher_list  = [], [], [], []
hang_state = True

for i in range(N) :
    row_corridor = list(map(str, input().split()))
    corridor.append(row_corridor)
    for j in range(N) :
        if row_corridor[j] == 'X' :
            blank_list.append((i,j))
        elif row_corridor[j] == 'S' :
            student_list.append((i,j))
        elif row_corridor[j] == 'T':
            teacher_list.append((i, j))


def start_monitor(corridor, row, colum):
    if row - 1 >= 0 and corridor[row - 1][colum] != 'T' :
        if corridor[row - 1][colum] == 'S':
            return False
        else :
            corridor[row - 1][colum] = 'T'
            if not start_monitor(corridor, row - 1, colum) :
                return False

    if row + 1 < N and corridor[row + 1][colum] != 'T' :
        if corridor[row + 1][colum] == 'S' :
            return False
        else :
            corridor[row + 1][colum] = 'T'
            if not start_monitor(corridor,row + 1, colum) :
                return False

    if colum - 1 >= 0 and corridor[row][colum - 1] != 'T' :
        if corridor[row][colum - 1] == 'S':
            return False
        else :
            corridor[row][colum - 1] = 'T'
            if not start_monitor(corridor, row, colum - 1) :
                return False
    if colum + 1 < N and corridor[row][colum + 1] != 'T' :
        if corridor[row][colum + 1] == 'S' :
            return False
        else :
            corridor[row][colum + 1] = 'T'
            if not start_monitor(corridor, row, colum + 1) :
                return False
    return True


def check_case() :
    for obstacle in combinations(blank_list, 3):
        corridor[obstacle[0][0]][obstacle[0][1]] = 'T'
        corridor[obstacle[1][0]][obstacle[1][1]] = 'T'
        corridor[obstacle[2][0]][obstacle[2][1]] = 'T'
        copy_corr = copy.deepcopy(corridor)

        for row, colum in teacher_list:
            hang_state = start_monitor(copy_corr, row, colum)
            if not hang_state:
                break

        if hang_state:
            return "Yes"
    return "No"

print(check_case())