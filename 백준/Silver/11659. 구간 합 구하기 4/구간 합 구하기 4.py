import sys

N, M = map(int, sys.stdin.readline().strip().split())
numbers = [0] + list(map(int, sys.stdin.readline().strip().split()))

for i in range(N) :
    numbers[i + 1] = numbers[i + 1] + numbers[i]

for _ in range(M):
    i, j = map(int, sys.stdin.readline().strip().split())
    print(numbers[j] - numbers[i - 1])
