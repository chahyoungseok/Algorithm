import sys

input1 = sys.stdin.readline().strip()
input2 = sys.stdin.readline().strip()

len_1, len_2 = len(input1), len(input2)

dp = [["" for _ in range(len_2 + 1)] for _ in range(len_1 + 1)]

for i in range(1, len_1 + 1) :
    for j in range(1, len_2 + 1) :
        if input1[i - 1] == input2[j - 1] :
            dp[i][j] = dp[i - 1][j - 1] + input1[i - 1]
        else :
            dp[i][j] = max(dp[i - 1][j], dp[i][j - 1], key=lambda x : len(x))

result = ""
for i in range(len_1 + 1) :
    result = max(result, max(dp[i], key=lambda x : len(x)), key=lambda x : len(x))
    
print(len(result))
if len(result) != 0 :
    print(result)