N, M = map(int, input().split())
a, b, d = map(int, input().split())

mapVis = [[0] * M for i in range(N)]
mapVis[a][b] = 1
visCount = 1

mapInf = []
for i in range(N) :
    mapInf.append(list(map(int, input().split())))

dirCount = 0
dx = [-1, 0, 1, 0]
dy = [0, 1, 0, -1]

while True :
    d = (d + 3) % 4
    CooA = a + dx[d]
    CooB = b + dy[d]

    if dirCount == 4 :
        invD = (d + 2) % 4
        a = a + dx[invD]
        b = b + dy[invD]

        if mapVis[a][b] == 1 or mapInf[a][b] == 1 :
            break
        else :
            mapVis[a][b] = 1
            visCount += 1
            continue

    if CooA < 0 or CooA > N - 1 or mapVis[CooA][CooB] == 1 or mapInf[CooA][CooB] == 1 :
        dirCount += 1
        continue
    else :
        a = CooA
        b = CooB
        mapVis[a][b] = 1
        visCount += 1
        dirCount = 0

print(visCount)