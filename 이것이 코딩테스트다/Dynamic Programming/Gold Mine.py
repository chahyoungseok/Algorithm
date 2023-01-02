T = int(input())
max_result = []
for cycle in range(T) :
    n, m = map(int, input().split())
    lists = list(map(int, input().split()))
    maps, max_maps = [], [[0] * n for _ in range(m)]

    for i in range(n):
        maps.append(lists[m * i: m * (i + 1)])
        max_maps[0][i] = lists[m * i]

    for i in range(1, m):
        for j in range(n):
            if j - 1 >= 0 and j + 1 < n:
                max_maps[i][j] = maps[j][i] + max(max_maps[i - 1][j - 1: j + 2])
            elif j - 1 < 0:
                max_maps[i][j] = maps[j][i] + max(max_maps[i - 1][j: j + 2])
            else:
                max_maps[i][j] = maps[j][i] + max(max_maps[i - 1][j - 1: j + 1])

    max_result.append(max(max_maps[m - 1]))

for result in max_result :
    print(result)