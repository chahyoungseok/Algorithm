import sys
from itertools import combinations

N, S = map(int, (sys.stdin.readline()).split())
N_list = list(map(int, (sys.stdin.readline()).split()))

count = 0
for i in range(1, N + 1) :
    for comb in combinations(N_list, i) :
        if S == sum(comb) :
            count += 1
print(count)
