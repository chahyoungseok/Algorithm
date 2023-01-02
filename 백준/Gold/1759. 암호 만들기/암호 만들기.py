from itertools import combinations

L, C = map(int, input().split())
a_list = list(map(str, input().split()))
a_list = sorted(a_list)

for combin in combinations(a_list, L) :
    m, j = 0, 0
    for i in combin :
        if i in ['a', 'e', 'i', 'o', 'u'] :
            m += 1
        else :
            j += 1

    if m > 0 and j > 1 :
        print("".join(combin))

