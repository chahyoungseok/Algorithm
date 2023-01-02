import sys, bisect

N = int(sys.stdin.readline().strip())
A = list(map(int, (sys.stdin.readline()).split()))

elements = [A[0]]
for i in range(1, N) :
    if A[i] > elements[-1] :
        elements.append(A[i])
    else :
        idx = bisect.bisect_left(elements, A[i])
        elements[idx] = A[i]
print(len(elements))