N = int(input())
K = int(input())

board = [[0] * N for _ in range(N)]
for i in range(K) :
    row, column = map(int, input().split())
    board[row - 1][column - 1] = 1

L = int(input())

direction = []
for i in range(L) :
    X, C = map(str, input().split())
    direction.append((X,C))

x, y, time, direct = 0, 0, 0, 2
mx = [-1,0,1,0]
my = [0,-1,0,1]

snake_body = [(0,0)]

while True :
    time += 1
    state = False

    x, y = x + mx[direct], y + my[direct]

    if x < 0 or y < 0 or x >= N or y >= N :
        print("wall break")
        break

    for fat in snake_body :
        if fat[0] == x and fat[1] == y :
            print("snake_body break")
            state = True
    if state :
        break

    snake_body.append((x,y))
    if board[y][x] != 1 :
        snake_body.pop(0)

    if direction and int(direction[0][0]) == time :
        C = direction[0][1]
        direction.pop(0)
        if C == 'L' :
            direct = (direct + 4 - 1) % 4
        else :
            direct = (direct + 4 + 1) % 4

print(time)