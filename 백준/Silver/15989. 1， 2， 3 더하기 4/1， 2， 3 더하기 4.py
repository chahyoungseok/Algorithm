import sys

T = int(input())
dp = [0] * 10001

for number in range(1, 4) :
    dp[number] += 1
    for i in range(1, 10001) :
        if 10001 > i + number :
            dp[i + number] += dp[i]

for _ in range(T) :
    print(dp[int(sys.stdin.readline().strip())])
