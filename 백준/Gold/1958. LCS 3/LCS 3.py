import sys

str1 = sys.stdin.readline().strip()
str2 = sys.stdin.readline().strip()
str3 = sys.stdin.readline().strip()

len_1, len_2, len_3 = len(str1), len(str2), len(str3)
dp = [[[0 for _ in range(len_3 + 1)] for _ in range(len_2 + 1)] for _ in range(len_1 + 1)]
for i in range(1, len_1 + 1) :
    for j in range(1, len_2 + 1) :
        for k in range(1, len_3 + 1) :
            if str1[i - 1] == str2[j - 1] and str2[j - 1] == str3[k - 1] :
                dp[i][j][k] = dp[i - 1][j - 1][k - 1] + 1
            else :
                dp[i][j][k] = max(dp[i - 1][j][k], dp[i][j - 1][k], dp[i][j][k - 1])

max_value = 0
for i in range(len_1 + 1) :
    for j in range(len_2 + 1) :
        max_value = max(max_value, max(dp[i][j]))

print(max_value)
