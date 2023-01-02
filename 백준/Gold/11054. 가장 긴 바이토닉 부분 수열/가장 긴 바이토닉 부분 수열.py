import sys

N = int(sys.stdin.readline().strip())
A = list(map(int, (sys.stdin.readline()).split()))

x = [1 for _ in range(N)]
y = [1 for _ in range(N)]
for i in range(N) :
    for j in range(i) :
        if A[i] > A[j] :
            x[i] = max(x[j] + 1, x[i])
        if A[N - i - 1] > A[N - j - 1]:
            y[N - i - 1] = max(y[N - j - 1] + 1, y[N - i - 1])

result = 0
for i in range(N) :
    result = max(result, x[i] + y[i] - 1)
print(result)