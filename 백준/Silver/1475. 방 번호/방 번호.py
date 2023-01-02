import sys

N = sys.stdin.readline().strip()
arr = [0 for _ in range(9)]

for char in N :
    target = int(char)
    if target == 6 or target == 9 :
        arr[6] += 1
    else :
        arr[int(char)] += 1

if arr[6] % 2 == 0 :
    arr[6] //= 2
else :
    arr[6] = arr[6] // 2 + 1

print(max(arr))