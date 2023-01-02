import sys

T = int(input())

for _ in range(T) :
    a, b = map(int, (sys.stdin.readline()).split())
    if a % 10 == 0 :
        print(10)
    else :
        if b % 4 == 0 :
            b = 4
        else :
            b %= 4
        print(((a % 10) ** b) % 10)