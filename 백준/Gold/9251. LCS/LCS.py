str_1 = input()
str_2 = input()

str_len_1 = len(str_1)
dp = [0] * str_len_1
for i in str_2 :
    pre_sel = 0
    for j in range(str_len_1) :
        if dp[j] > pre_sel:
            pre_sel = dp[j]
        elif str_1[j] == i :
            dp[j] = pre_sel + 1
        
print(max(dp))