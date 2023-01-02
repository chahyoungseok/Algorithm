from collections import deque


def solution(board):
    q, N = deque(), len(board)
    q.append([[0, 0], [0, 1], 0, 0])
    dx1, dy1 = [1, -1, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1], [0, 0, 1, -1, 1, 1, 0, 0, 0, 0, 1, -1]
    dx2, dy2 = [1, -1, 0, 0, 0, 0, 1, -1, -1, -1, 0, 0], [0, 0, 1, -1, 0, 0, -1, -1, 1, -1, 0, 0]

    visited = [[[True, True] for _ in range(N)] for _ in range(N)]
    visited[0][1][0] = False

    # state 0 : 가로상태, 1 : 세로상태
    while q:
        p1, p2, state, dist = q.popleft()
        # print("current", p1, p2, dist, state)
        if p2[0] == N - 1 and p2[1] == N - 1:
            return dist

        for i in range(12):
            if (state == 1 and (4 <= i <= 7)) or (state == 0 and (8 <= i <= 11)):
                continue
            mx1, my1, mx2, my2 = p1[0] + dx1[i], p1[1] + dy1[i], p2[0] + dx2[i], p2[1] + dy2[i]

            if 0 <= mx1 < N and 0 <= mx2 < N and 0 <= my1 < N and 0 <= my2 < N and board[mx1][my1] == 0 and board[mx2][my2] == 0:
                # print("--------------------------------------")
                # print(mx1, my1, mx2, my2)
                # print("p1, p2 ",p1, p2)
                # print("i", i)
                if i > 3:
                    if (4 <= i <= 5) and (board[mx1][p1[1]] == 1):
                        continue
                    if (6 <= i <= 7) and (board[mx2][p2[1]] == 1):
                        continue
                    if (8 <= i <= 9) and (board[p2[0]][my2] == 1):
                        continue
                    if (10 <= i <= 11) and (board[p1[0]][my1] == 1):
                        continue

                copy_state = state
                if i >= 8:
                    copy_state = 0
                elif i >= 4:
                    copy_state = 1

                p_list = [[mx1, my1], [mx2, my2]]
                p_list = sorted(p_list, key=lambda x: (x[0], x[1]))

                if not visited[p_list[1][0]][p_list[1][1]][copy_state]:
                    continue

                # print(p_list)

                visited[p_list[1][0]][p_list[1][1]][copy_state] = False
                q.append([p_list[0], p_list[1], copy_state, dist + 1])
