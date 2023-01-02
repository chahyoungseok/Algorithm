import sys

N = int(sys.stdin.readline().strip())
A = list(map(int, (sys.stdin.readline()).split()))
A_hat = sorted(list(set(A)))
A_dict = {}
for i in range(len(A_hat)) :
    A_dict[A_hat[i]] = i

for i in range(N) :
    print(A_dict[A[i]], end=" ")
