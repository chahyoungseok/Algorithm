import sys

N = int(input())
stack = []
for _ in range(N) :
    order = list(map(str, sys.stdin.readline().split()))
    if order[0] == "push" :
        stack.append(order[1])
    elif order[0] == "pop" :
        s_len = len(stack)
        if s_len != 0:
            print(stack.pop(s_len - 1))
        else:
            print(-1)
    elif order[0] == "size" :
        print(len(stack))
    elif order[0] == "empty" :
        if len(stack) == 0:
            print(1)
        else:
            print(0)
    elif order[0] == "top" :
        s_len = len(stack)
        if s_len == 0:
            print(-1)
        else:
            print(stack[s_len - 1])