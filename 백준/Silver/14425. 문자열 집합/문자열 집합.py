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


N, M = map(int, input().split())
S = []
for _ in range(N) :
    S.append(sys.stdin.readline())

S_len, total = len(S), 0
S = sorted(S)
for _ in range(M) :
    data = sys.stdin.readline()
    if binary_search(S, data, 0, S_len - 1) != -1 :
        total += 1

print(total)