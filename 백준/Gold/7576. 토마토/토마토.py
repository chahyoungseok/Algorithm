import sys

M, N = map(int, input().split())

board, state, board_state = [], False, True
q, temp_q = [], []
for i in range(N) :
    data = list(map(int, (sys.stdin.readline()).split()))
    for j in range(M) :
        if data[j] == 1 :
            q.append([i,j])
    board.append(data)
    if not state and 0 in data :
        state = True

dx = [1, 0, -1, 0]
dy = [0, 1, 0, -1]

if state :
    days, em = 0, True
    while q and em:
        days += 1
        em = False
        for x, y in q:
            for i in range(4):
                mx, my = x + dx[i], y + dy[i]
                if mx >= 0 and mx < N and my >= 0 and my < M:
                    if board[mx][my] == 0:
                        em = True
                        board[mx][my] = 1
                        temp_q.append([mx, my])
        q = temp_q
        temp_q = []

    for i in range(N) :
        if 0 in board[i] :
            board_state = False

    if board_state :
        print(days - 1)
    else :
        print(-1)

else :
    print(0)