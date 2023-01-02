import sys

N = int(input())
A = list(map(int, (sys.stdin.readline()).split()))
B = list(map(int, (sys.stdin.readline()).split()))

A = sorted(A)
B = sorted(B, reverse=True)

total = 0
for i in range(N) :
    total += A[i] * B[i]

print(total)