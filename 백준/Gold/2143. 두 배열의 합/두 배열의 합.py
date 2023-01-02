import sys, bisect

T = int(sys.stdin.readline().strip())
n = int(sys.stdin.readline().strip())
A = list(map(int, (sys.stdin.readline()).split()))
m = int(sys.stdin.readline().strip())
B = list(map(int, (sys.stdin.readline()).split()))

a_case, b_case = [], []
for i in range(n) :
    result = 0
    for j in range(i, n) :
        result += A[j]
        a_case.append(result)

for i in range(m) :
    result = 0
    for j in range(i, m) :
        result += B[j]
        b_case.append(result)

a_case = sorted(a_case)
total = 0
for i in b_case :
    total += bisect.bisect_right(a_case, T - i) - bisect.bisect_left(a_case, T - i)

print(total)