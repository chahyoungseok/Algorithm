from collections import deque

N, M = map(int, input().split())

mazeMap = []
for i in range(N) :
    mazeMap.append(list(map(int, input())))

x, y = 0, 0

dx = [-1, 0, 1, 0]
dy = [0, -1, 0, 1]

mazeQueue = deque()
mazeQueue.append((x, y))

while mazeQueue :
    x, y = mazeQueue.popleft()

    for i in range(4) :
        preX = dx[i] + x
        preY = dy[i] + y

        if preX < 0 or preY < 0 or preX > N - 1 or preY > M - 1 :
            continue

        if mazeMap[preX][preY] == 1 :
            mazeQueue.append((preX, preY))
            mazeMap[preX][preY] = mazeMap[x][y] + 1

print(mazeMap[N - 1][M - 1])