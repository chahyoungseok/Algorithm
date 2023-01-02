import sys

A, B, C = map(int, (sys.stdin.readline()).split())


def mu(a, b) :
    if b == 1 :
        return a % C
    elif b == 0 :
        return 1 % C

    if b % 2 == 0 :
        return (mu(a, b // 2) ** 2) % C
    else :
        return (mu(a, b // 2) ** 2 * a) % C


print(mu(A, B))