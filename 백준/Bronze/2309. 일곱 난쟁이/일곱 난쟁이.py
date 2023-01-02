import sys
from itertools import combinations

height = []
for _ in range(9) :
    height.append(int(sys.stdin.readline()))

for combin in combinations(height, 7) :
    if sum(list(combin)) == 100 :
        r = sorted(list(combin))
        for i in r :
            print(i)
        break