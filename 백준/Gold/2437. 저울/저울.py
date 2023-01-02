import sys

N = int(sys.stdin.readline().strip())
arr = sorted(list(map(int, (sys.stdin.readline()).split())))
save_max = arr[0]
if arr[0] != 1 :
    print(1)
else :
    for i in range(1, N) :
        if arr[i] > save_max + 1 :
            break
        else :
            save_max = save_max + arr[i]
    print(save_max + 1)

