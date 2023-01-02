A = input()
B = input()

A_len, B_len, state = len(A), len(B), False
dp = [[0] * B_len for _ in range(A_len)]

for i in range(A_len) :
    dp[i][0] = i + 1
    if state or B[0] == A[i] :
        state = True
        dp[i][0] -= 1

state = False
for i in range(B_len) :
    dp[0][i] = i + 1
    if state or A[0] == B[i] :
        state = True
        dp[0][i] -= 1


for i in range(1, A_len) :
    for j in range(1, B_len) :
        if A[i] == B[j] :
            dp[i][j] = dp[i - 1][j - 1]
        else :
            dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])

print(dp[A_len - 1][B_len - 1])