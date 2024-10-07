import sys

M = int(sys.stdin.readline().strip())
S = 0
for _ in range(M):
    data = list(map(str, sys.stdin.readline().strip().split()))
    if data[0] == "add":
        S |= 1 << int(data[1])
    elif data[0] == "remove":
        S &= ~(1 << int(data[1]))
    elif data[0] == "check":
        if S & (1 << int(data[1])) == 0:
            print(0)
        else:
            print(1)
    elif data[0] == "toggle":
        if S & (1 << int(data[1])) == 0:
            S |= (1 << int(data[1]))
        else:
            S &= ~(1 << int(data[1]))
    elif data[0] == "all":
        S = 2**21 - 1
    elif data[0] == "empty":
        S = 0

