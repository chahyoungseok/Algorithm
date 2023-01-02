import sys

pre_mid = 0


def binary_search(target, start, end) :
    global pre_mid
    if start > end :
        return -1
    mid = (start + end) // 2
    total = 0
    for i in trees :
        if i > mid :
            total += i - mid
    if total >= target :
        pre_mid = mid
        return binary_search(target, mid + 1, end)
    else :
        return binary_search(target, start, mid - 1)


N, M = map(int, input().split())
trees = list(map(int, (sys.stdin.readline()).split()))
if M == 0 :
    print(0)
else :
    binary_search(M, 0, 1000000000)
    print(pre_mid)