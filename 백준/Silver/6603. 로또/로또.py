import sys
from itertools import combinations

while True :
    data = list(map(int, (sys.stdin.readline()).split()))
    if data == [0] :
        break

    for comb in combinations(data[1:], 6) :
        for i in comb :
            print(i, end=" ")
        print()
    print()