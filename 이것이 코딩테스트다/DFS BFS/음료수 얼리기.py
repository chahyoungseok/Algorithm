def confirmAround(i, j) :
    global iceMap

    if i < 0 or i > len(iceMap) - 1 or j < 0 or j > len(iceMap[0]) - 1  :
        return False

    if iceMap[i][j] == 0 :
        iceMap[i][j] = 1

        confirmAround(i - 1, j)
        confirmAround(i, j + 1)
        confirmAround(i + 1, j)
        confirmAround(i, j - 1)

        return True
    return False


N, M = map(int, input().split())

iceMap = []
iceCount = 0

for i in range(N) :
    iceMap.append(list(map(int,input())))


for i in range(N) :
    for j in range(M) :
        if confirmAround(i, j) :
            iceCount += 1

print(iceCount)