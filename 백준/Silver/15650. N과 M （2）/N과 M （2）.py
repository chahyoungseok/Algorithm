from itertools import combinations

N, M = map(int, input().split())

for case in combinations(range(1, N + 1), M) :
    for i in case :
        print(i, end=" ")
    print()