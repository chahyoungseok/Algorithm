import sys

n = int(sys.stdin.readline().strip())
ju, dp = [], [0] * n
for _ in range(n) :
    ju.append(int(sys.stdin.readline().strip()))

if n <= 2 :
    print(sum(ju))
elif n == 3 :
    print(max(ju[0] + ju[2], ju[1] + ju[2], ju[0] + ju[1]))
else :
    dp[0], dp[1] = ju[0], ju[0] + ju[1]
    dp[2] = max(ju[0] + ju[2], ju[1] + ju[2])
    dp[3] = max(ju[3] + ju[2] + ju[0], ju[3] + ju[1] + ju[0])

    for i in range(4, n) :
        dp[i] = ju[i] + max(dp[i - 2], ju[i - 1] + dp[i - 3], ju[i - 1] + dp[i - 4])

    print(max(dp))