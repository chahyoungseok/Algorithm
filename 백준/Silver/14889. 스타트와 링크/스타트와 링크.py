import sys
from itertools import combinations

N = int(sys.stdin.readline().strip())

board, N_list = [], [i for i in range(N)]
min_value = int(1e11)
for _ in range(N) :
    board.append(list(map(int, sys.stdin.readline().split())))

for comb in combinations(N_list, N // 2) :
    sum_value_1 = 0
    for c in comb :
        for cc in comb :
            sum_value_1 += board[c][cc]

    comb_2 = []
    for i in N_list :
        if i not in comb :
            comb_2.append(i)

    sum_value_2 = 0
    for c in comb_2:
        for cc in comb_2:
            sum_value_2 += board[c][cc]

    min_value = min(min_value, abs(sum_value_2 - sum_value_1))

print(min_value)