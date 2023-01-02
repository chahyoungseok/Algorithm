import copy

dp = [[] for _ in range(12)]
dp[1].append([1])
dp[2].append([2])
dp[3].append([3])
q, temp = [[1], [2], [3]], []

for _ in range(11) :
    for i in q :
        for j in range(1, 4) :
            c = sum(i) + j
            if 12 > c :
                i_copy = copy.deepcopy(i)
                i_copy.append(j)
                if i_copy not in dp[c] :
                    dp[c].append(i_copy)
                    temp.append(i_copy)
    q = copy.deepcopy(temp)
    
T = int(input())
for _ in range(T) :
    print(len(dp[int(input())]))