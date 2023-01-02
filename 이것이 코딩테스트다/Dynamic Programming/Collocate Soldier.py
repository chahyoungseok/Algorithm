N = int(input())
soldier_list = list(map(int,input().split()))

dp = [1] * N

for i in range(N) :
    for j in range(i) :
        if soldier_list[i] < soldier_list[j] :
            dp[i] = max(dp[i], dp[j] + 1)

print(N - max(dp))