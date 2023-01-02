import sys

T = int(sys.stdin.readline().strip())


def gcd(n, m) :
    mod = m % n
    if mod != 0 :
        m, n = n, mod
        return gcd(n, m)
    else:
        return n


def lcm(n, m) :
    return int(n*m / gcd(n, m))


for _ in range(T) :
    M, N, x, y = map(int, (sys.stdin.readline()).split())
    lcm_mn = lcm(M, N)
    day, total, state = 0, 0, True
    if x > y :
        sub = x - y
        while sub != day :
            day = (day + N) % M
            total += N
            if total > lcm_mn :
                state = False
                break

        if state :
            print(total + y)
        else :
            print(-1)
    else :
        sub = y - x
        while sub != day :
            day = (day + M) % N
            total += M
            if total > lcm_mn :
                state = False
                break

        if state:
            print(total + x)
        else:
            print(-1)
