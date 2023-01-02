import sys

n, k = map(int, input().split())
coins = []
for _ in range(n) :
    coin = int(sys.stdin.readline().strip())
    if 10001 > coin :
        coins.append(coin)

coins = sorted(coins)
dp = [0] * 10001

for coin in coins :
    dp[coin] += 1
    for i in range(1, 10001 - coin) :
        if dp[i] != 0 :
            dp[i + coin] += dp[i]

print(dp[k])
