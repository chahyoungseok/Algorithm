from itertools import combinations
import copy

def check_zero(maps) :
    zero_sum = 0
    for i in range(len(maps)) :
        for j in range(len(maps[0])) :
            if maps[i][j] == 0 :
                zero_sum += 1

    return zero_sum

def dfs(maps, row, colum) :
    if row - 1 >= 0 and maps[row - 1][colum] == 0:
        maps[row - 1][colum] = 2
        dfs(maps, row - 1, colum)
    if row + 1 < len(maps) and maps[row + 1][colum] == 0:
        maps[row + 1][colum] = 2
        dfs(maps, row + 1, colum)
    if colum - 1 >= 0 and maps[row][colum - 1] == 0:
        maps[row][colum - 1] = 2
        dfs(maps, row, colum - 1)
    if colum + 1 < len(maps[0]) and maps[row][colum + 1] == 0:
        maps[row][colum + 1] = 2
        dfs(maps, row, colum + 1)

    return maps


N, M = map(int, input().split())
maps, wall_list, virus_list = [], [], []
max_zero = 0

for i in range(N) :
    keep = list(map(int, input().split()))
    maps.append(keep)
    for j in range(M) :
        if keep[j] == 0 :
            wall_list.append((i,j))
        elif keep[j] == 2 :
            virus_list.append((i,j))

for walls in combinations(wall_list,3) :
    copy_map = copy.deepcopy(maps)
    copy_map[walls[0][0]][walls[0][1]] = 1
    copy_map[walls[1][0]][walls[1][1]] = 1
    copy_map[walls[2][0]][walls[2][1]] = 1

    for row, colum in virus_list:
        copy_map = dfs(copy_map, row, colum)

    max_zero = max(check_zero(copy_map), max_zero)

print(max_zero)