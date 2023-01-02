from collections import deque

N = int(input())
board = []
for _ in range(N) :
    board.append(list(input()))

cases = []
dx, dy = [1, 0, -1, 0], [0, 1, 0, -1]
for i in range(N) :
    for j in range(N) :
        if board[i][j] == '1' :
            q, case = deque(), set()
            q.append([i,j])

            while q :
                x, y = q.popleft()
                case.add((x, y))
                board[x][y] = '0'

                for k in range(4) :
                    mx, my = x + dx[k], y + dy[k]
                    if mx >=0 and mx < N and my >= 0 and my < N :
                        if board[mx][my] == '1' and [mx,my] not in q:
                            q.append([mx, my])

            cases.append(len(case))

print(len(cases))
cases = sorted(cases)
for i in cases :
    print(i)