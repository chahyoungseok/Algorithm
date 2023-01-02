import sys
from collections import deque

T = int(input())
for _ in range(T) :
    state_error, state_reverse = False, False
    p = sys.stdin.readline().strip()
    n = int(sys.stdin.readline())
    arr = input()
    if arr == "[]" :
        arr = deque()
    else :
        arr = deque(list(map(int, arr[1:-1].split(","))))

    for i in p :
        if arr :
            if i == "R" :
                if state_reverse :
                    state_reverse = False
                else :
                    state_reverse = True
            elif i == "D" :
                if state_reverse :
                    arr.pop()
                else :
                    arr.popleft()
        else :
            if i == "D":
                state_error = True
                print("error")
                break

    if not state_error :
        if state_reverse :
            arr.reverse()

        arr_len = len(arr)
        print("[", end="")
        for i in range(arr_len):
            if i != arr_len - 1:
                print(arr[i], end=",")
            else:
                print(arr[i], end="")
        print("]")
