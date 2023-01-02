import sys

T = int(input())
for _ in range(T) :
    N = int(input())
    e = []
    for _ in range(N) :
        e.append(list(map(int, (sys.stdin.readline()).split())))

    e = sorted(e, key=lambda x : x[0])
    total, standard = 1, e[0][1]

    for i in range(1, N) :
        if standard > e[i][1] :
            total += 1
            standard = e[i][1]

    print(total)