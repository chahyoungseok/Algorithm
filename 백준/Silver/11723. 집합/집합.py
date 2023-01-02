import sys


M = int(input())
S = 0
for _ in range(M) :
    data = list(map(str, sys.stdin.readline().split()))
    if data[0] == "add" :
        if S & (1 << int(data[1])) == 0 :
            S |= (1 << int(data[1]))
    elif data[0] == "remove" :
        if S & (1 << int(data[1])) != 0 :
            S &= ~(1 << int(data[1]))
    elif data[0] == "check" :
        if S & (1 << int(data[1])) != 0:
            print(1)
        else :
            print(0)
    elif data[0] == "toggle":
        if S & (1 << int(data[1])) != 0:
            S &= ~(1 << int(data[1]))
        else :
            S |= (1 << int(data[1]))
    elif data[0] == "all":
        S = (1 << 21) - 1
    else :
        S = 0