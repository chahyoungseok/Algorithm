import sys

A = (sys.stdin.readline().strip())
B = (sys.stdin.readline().strip())

A_len, B_len = len(A), len(B)
dp = [[0 for _ in range(B_len)] for _ in range(A_len)]

max_count = 0

for i in range(A_len) :
    for j in range(B_len) :
        if A[i] == B[j] :
            if i == 0 or j == 0 :
                dp[i][j] = 1
            else :
                dp[i][j] = dp[i - 1][j - 1] + 1
            max_count = max(max_count, dp[i][j])

print(max_count)