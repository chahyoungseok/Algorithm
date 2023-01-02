import copy
import sys

N, M, K = map(int, (sys.stdin.readline()).split())
A, board = [], [[[[], 5] for _ in range(N)] for _ in range(N)]

for _ in range(N) :
    A.append(list(map(int, (sys.stdin.readline()).split())))

for _ in range(M) :
    x, y, z = map(int, (sys.stdin.readline()).split())
    board[x - 1][y - 1][0].append(z)

dx, dy = [1, -1, 0, 0, -1, 1, 1, -1], [0, 0, 1, -1, -1, 1, -1, 1]

for _ in range(K) :
    # 봄, 여름
    for i in range(N) :
        for j in range(N) :
            if board[i][j][0] :
                target, food = sorted(board[i][j][0]), board[i][j][1]

                k, add_food, target_len, state = 0, 0, len(target), True
                for k in range(target_len) :
                    food -= target[k]
                    if food < 0 :
                        food += target[k]
                        state = False
                        break
                    target[k] += 1
                if not state :
                    for d in range(k, target_len) :
                        add_food += target[d] // 2
                    k -= 1
                board[i][j][0] = copy.deepcopy(target[:k + 1])
                board[i][j][1] = food + add_food

    # 가을, 겨울
    for i in range(N) :
        for j in range(N) :
            if board[i][j][0] :
                target = board[i][j][0]
                target_len = len(target)
                for k in range(target_len) :
                    if target[k] % 5 == 0 :
                        for s in range(8) :
                            mx, my = i + dx[s], j + dy[s]
                            if 0 <= mx < N and 0 <= my < N :
                                board[mx][my][0].append(1)
            board[i][j][1] += A[i][j]
            
result = 0
for i in range(N) :
    for j in range(N) :
        if board[i][j] :
            result += len(board[i][j][0])

print(result)