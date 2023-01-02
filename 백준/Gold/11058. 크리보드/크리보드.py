N = int(input())
dp = [0] * 101
dp[1] = 1

for i in range(1, 101) :
    index = 2
    for j in range(index + i, 101) :
        if 101 > j + 1 :
            dp[j + 1] = max(dp[j + 1], dp[i] * index)
            index += 1

    index = 0
    for j in range(i, 101) :
        if 101 > index + j + 1 :
            dp[index + j + 1] = max(dp[index + j + 1], dp[i] + index + 1)
            index += 1
print(dp[N])