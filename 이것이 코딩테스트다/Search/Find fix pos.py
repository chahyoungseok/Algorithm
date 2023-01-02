def binary_search(arr, start, end) :
    mid = (start + end) // 2

    if start > end :
        return None

    if arr[mid] > mid :
        return binary_search(arr, start, mid - 1)
    elif arr[mid] < mid :
        return binary_search(arr, mid + 1, end)
    else :
        return mid

N = int(input())
arr = list(map(int, input().split()))

print(binary_search(arr, 0, N - 1))