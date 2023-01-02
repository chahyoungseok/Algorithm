import sys
from itertools import permutations
from collections import deque

N = int(sys.stdin.readline().strip())
scv_list = list(map(int, (sys.stdin.readline()).split()))

if N == 1 :
    value = scv_list[0]
    result = value // 9
    if value % 9 != 0 :
        result += 1
    print(result)
elif N == 2 :
    scv_list.append(0)
    q = deque()
    q.append(scv_list)

    while q :
        scv_a, scv_b, dist = q.popleft()

        if scv_a == 0 and scv_b == 0 :
            print(dist)
            break

        for a, b in permutations([9, 3], 2) :
            next_scv = []
            if a >= scv_a :
                next_scv.append(0)
            else :
                next_scv.append(scv_a - a)

            if b >= scv_b :
                next_scv.append(0)
            else :
                next_scv.append(scv_b - b)

            next_scv.append(dist + 1)
            q.append(next_scv)
else :
    scv_list.append(0)
    q = deque()
    q.append(scv_list)

    while q:
        scv_a, scv_b, scv_c, dist = q.popleft()

        if scv_a == 0 and scv_b == 0 and scv_c == 0:
            print(dist)
            break

        for a, b, c in permutations([9, 3, 1], 3):
            next_scv = []
            if a >= scv_a:
                next_scv.append(0)
            else:
                next_scv.append(scv_a - a)

            if b >= scv_b:
                next_scv.append(0)
            else:
                next_scv.append(scv_b - b)

            if c >= scv_c:
                next_scv.append(0)
            else:
                next_scv.append(scv_c - c)

            next_scv.append(dist + 1)

            if next_scv not in q :
                q.append(next_scv)

