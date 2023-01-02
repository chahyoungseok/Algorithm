import sys


def binary_search(arr, target, start, end) :
    if start > end :
        return -1
    mid = (start + end) // 2

    if arr[mid] > target :
        return binary_search(arr, target, start, mid - 1)
    elif arr[mid] < target :
        return binary_search(arr, target, mid + 1, end)
    else :
        return mid


N = int(input())
N_list = list(map(int, (sys.stdin.readline()).split()))
N_list, N_list_len = sorted(N_list), len(N_list)

M = int(input())
M_list = list(map(int, (sys.stdin.readline()).split()))

for i in M_list :
    if binary_search(N_list, i, 0, N_list_len - 1) != -1 :
        print(1)
    else :
        print(0)