from itertools import permutations

N, M = map(int, input().split())

for case in permutations(range(1, N + 1), M) :
    for i in case :
        print(i, end=" ")
    print()