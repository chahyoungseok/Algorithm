import sys

N, C = map(int, (sys.stdin.readline()).split())
x, answer = [], 1
for _ in range(N) :
    x.append(int(sys.stdin.readline().strip()))
x = sorted(x)


def binary_search(arr, start, end) :
    if start > end :
        return
    mid = (start + end) // 2

    current, count = arr[0], 1
    for i in range(1, N) :
        if arr[i] >= current + mid :
            current = arr[i]
            count += 1
            
    global answer
    if count >= C :
        answer = mid
        binary_search(arr, mid + 1, end)
    else :
        binary_search(arr, start, mid - 1)


binary_search(x, 1, x[-1] - x[0])
print(answer)