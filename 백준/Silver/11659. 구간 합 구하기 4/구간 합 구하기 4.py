import sys

input = sys.stdin.readline
N, M = map(int, input().split())
board = list(map(int, input().split()))

start_arr = [0] * (N + 1)
for i in range(N) :
    start_arr[i + 1] = start_arr[i] + board[i]

for _ in range(M) :
    start, end = map(int, input().split())
    print(start_arr[end] - start_arr[start - 1])
