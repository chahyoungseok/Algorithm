import math, sys


def catalan(number) :
    standard = math.factorial(number)
    return math.factorial(number * 2) // (standard ** 2 * (number + 1))


T = int(sys.stdin.readline().strip())

for _ in range(T) :
    L = int(sys.stdin.readline().strip())
    if L % 2 == 0 :
        print(catalan(L // 2) % 1000000007)
    else :
        print(0)