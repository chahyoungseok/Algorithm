import sys

N = int(sys.stdin.readline().strip())
k = int(sys.stdin.readline().strip())


def binary_search(start, end) :
    if start > end :
        return

    mid = (start + end) // 2

    count = 0
    for i in range(1, N + 1) :
        target = mid // i
        if target > N :
            target = N
        count += target

    if count >= k :
        global result
        result = mid
        binary_search(start, mid - 1)
    else :
        binary_search(mid + 1, end)


result = 0
binary_search(1, min(10 ** 9, N ** 2))
print(result)