import sys

N = int(sys.stdin.readline().strip())
A = list(map(int, (sys.stdin.readline()).split()))

result = [-1 for _ in range(N)]
tmp = []
for i in range(N) :
    while tmp and A[tmp[-1]] < A[i] :
        result[tmp.pop()] = A[i]
    tmp.append(i)

print(*result)