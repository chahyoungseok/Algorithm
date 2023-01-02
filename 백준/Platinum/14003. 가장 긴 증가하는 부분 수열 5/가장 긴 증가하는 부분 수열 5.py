import sys, bisect

N = int(sys.stdin.readline().strip())
A = list(map(int, (sys.stdin.readline()).split()))
elements = [A[0]]
dp = [1 for _ in range(N)]

for i in range(1, N) :
    if A[i] > elements[-1] :
        elements.append(A[i])
        dp[i] = len(elements)
    else :
        idx = bisect.bisect_left(elements, A[i])
        elements[idx] = A[i]
        dp[i] = idx + 1

elements_len = len(elements)
print(elements_len)
ans = []

for i in range(N - 1, -1, -1) :
    if dp[i] == elements_len :
        ans.append(A[i])
        elements_len -= 1

    if elements_len < 1 :
        break

print(*ans[::-1])