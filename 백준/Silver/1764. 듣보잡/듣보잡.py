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


N, M = map(int, (sys.stdin.readline()).split())
N_list, result = [], []
for _ in range(N) :
    N_list.append((sys.stdin.readline()).strip())

N_list, total = sorted(N_list), 0
for _ in range(M) :
    data = (sys.stdin.readline()).strip()
    if binary_search(N_list, data, 0, N - 1) != - 1 :
        total += 1
        result.append(data)

result = sorted(result)
print(total)
for i in result :
    print(i)