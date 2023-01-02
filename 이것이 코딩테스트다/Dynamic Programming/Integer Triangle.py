n = int(input())

tri = []
max_tri = [[0] * i for i in range(1,n + 1)]

for _ in range(n) :
    tri.append(list(map(int, input().split())))

max_tri[0][0] = tri[0][0]

for i in range(1, n) :
    for j in range(i + 1) :
        if j - 1 >= 0 and j < i :
            max_tri[i][j] = tri[i][j] + max(max_tri[i - 1][j], max_tri[i - 1][j - 1])
        elif j - 1 < 0 :
            max_tri[i][j] = tri[i][j] + max_tri[i - 1][j]
        else :
            max_tri[i][j] = tri[i][j] + max_tri[i - 1][j - 1]

print(max(max_tri[n - 1]))