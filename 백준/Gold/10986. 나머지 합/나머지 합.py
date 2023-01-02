import sys

N, M = map(int, input().split())
board = list(map(int, sys.stdin.readline().split()))
total, result = 0, 0
arr = [0] * (M)
for i in range(N) :
    total += board[i]
    remain = total % M
    if remain == 0 :
        result += 1
    arr[remain] += 1

for i in range(M) :
    if arr[i] == 0:
        continue
    result += arr[i] * (arr[i] - 1) // 2

print(result)