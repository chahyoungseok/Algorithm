import sys

n = int(input())
n_list = list(map(int, (sys.stdin.readline()).split()))
dp = [0] * n
dp[0] = n_list[0]

for i in range(1, n) :
    if i > 0 :
        dp[i] = max(dp[i - 1] + n_list[i], n_list[i])
    else :
        dp[i] = n_list[i]

print(max(dp))