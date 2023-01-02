import sys

TC = int(sys.stdin.readline().strip())
for _ in range(TC) :
    N, M, W = map(int, (sys.stdin.readline()).split())

    edges = [[int(1e9) for _ in range(N + 1)] for _ in range(N + 1)]
    for _ in range(M) :
        S, E, T = map(int, (sys.stdin.readline()).split())
        edges[S][E] = min(edges[S][E], T)
        edges[E][S] = min(edges[E][S], T)

    for _ in range(W) :
        S, E, T = map(int, (sys.stdin.readline()).split())
        edges[S][E] = min(edges[S][E], -T)


    def floyd() :
        for k in range(1, N + 1) :
            for i in range(1, N + 1) :
                for j in range(1, N + 1) :
                    target = edges[i][k] + edges[k][j]
                    if target >= edges[i][j] :
                        continue
                    edges[i][j] = target
                    if target + edges[j][i] < 0 :
                        return True
        return False

    if floyd() :
        print("YES")
    else :
        print("NO")
