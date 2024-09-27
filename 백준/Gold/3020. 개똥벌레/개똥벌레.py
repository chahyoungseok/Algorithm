import sys


def binary_search(arr, target, start, end):
    mid = (start + end) // 2
    if start > end:
        return mid

    if arr[mid] >= target:
        return binary_search(arr, target, start, mid - 1)
    else :
        return binary_search(arr, target, mid + 1, end)


N, H = map(int, sys.stdin.readline().strip().split())

stalagmite, stalactite = [], []
for _ in range(N // 2):
    stalagmite.append(int(sys.stdin.readline().strip()))
    stalactite.append(int(sys.stdin.readline().strip()))

stalagmite, stalactite = sorted(stalagmite), sorted(stalactite)
obstacle_len = N // 2

min_value = sys.maxsize
count = 0
for height in range(1, H + 1):
    stalagmite_hit_count = obstacle_len - binary_search(stalagmite, height, 0, obstacle_len - 1) - 1
    stalactite_hit_count = obstacle_len - binary_search(stalactite, (H - height + 1), 0, obstacle_len - 1) - 1
    value = stalagmite_hit_count + stalactite_hit_count
    if min_value > value:
        min_value = value
        count = 1
    elif min_value == value:
        count += 1

print(str(min_value) + " " + str(count))