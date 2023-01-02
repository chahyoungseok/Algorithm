N = int(input())
consulting = []
max_consulting = [0] * (N + 1)

for _ in range(N) :
    consulting.append(list(map(int, input().split())))

for i in range(N) :
    target_index = consulting[i][0] + i
    if target_index < N + 1 :
        for j in range(target_index, N + 1) :
            max_consulting[target_index] = max(consulting[i][1] + max_consulting[i], max_consulting[target_index])
            if max_consulting[j] < max_consulting[target_index] :
                max_consulting[j] = max_consulting[target_index]


print(max(max_consulting))