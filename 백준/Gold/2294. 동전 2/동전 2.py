import sys

n, k = map(int, input().split())
coins = []
for _ in range(n) :
    coin = int(sys.stdin.readline().strip())
    if 10001 > coin :
        coins.append(coin)

coins = sorted(coins)
dp = [int(1e9)] * 10001

for coin in coins :
    dp[coin] = 1
    for i in range(1, 10001 - coin) :
        if dp[i] != int(1e9) :
            dp[i + coin] = min(dp[i] + 1, dp[i + coin])

if dp[k] != int(1e9):
    print(dp[k])
else :
    print(-1)