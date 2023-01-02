import sys


def binary_search(arr, target, start, end, standard) :
    global state
    if start > end :
        return -1
    mid = (start + end) // 2

    if standard - arr[mid] >= target :
        state = mid
        return binary_search(arr, target, mid + 1, end, standard)
    elif standard - arr[mid] < target :
        return binary_search(arr, target, start, mid - 1, standard)


N, S = map(int, input().split())
N_list = list(map(int, (sys.stdin.readline()).split()))
continuous, min_sel = [0], int(1e9)
state = -1

for i in range(N) :
    continuous.append(N_list[i] + continuous[i])

if S > continuous[N] :
    print(0)
else :
    for i in range(N + 1) :
        state = -1
        if S > continuous[i] :
            continue

        binary_search(continuous, S, 0, N, continuous[i])
        if state != -1 :
            min_sel = min(min_sel, i - state)

    print(min_sel)
