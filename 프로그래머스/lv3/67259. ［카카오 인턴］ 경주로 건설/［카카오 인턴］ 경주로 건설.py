from collections import deque


# direction : 0은 가로이동 중, 1은 세로이동 중
def solution(board):
    N, min_cost = len(board), int(1e9)

    dp = [[[int(1e9), int(1e9)] for _ in range(N)] for _ in range(N)]
    visited = [[True for _ in range(N)] for _ in range(N)]
    visited[0][0] = False

    dx, dy = [1, -1, 0, 0], [0, 0, 1, -1]

    q = deque()
    q.append([0, 0, 0, 0])

    while q :
        x, y, cost, direction = q.popleft()

        if cost >= dp[x][y][direction] :
            continue

        dp[x][y][direction] = cost

        if x == N - 1 and y == N - 1:
            min_cost = min(min_cost, cost)
            continue

        for i in range(4):
            mx, my = x + dx[i], y + dy[i]
            if 0 <= mx < N and 0 <= my < N and board[mx][my] == 0 and visited[mx][my] :
                visited[mx][my] = False

                if i > 1:
                    current_direction = 0
                else:
                    current_direction = 1

                if direction == current_direction:
                    q.append([mx, my, cost + 100, current_direction])
                else:
                    q.append([mx, my, cost + 600, current_direction])

                visited[mx][my] = True

    q.append([0, 0, 0, 1])

    while q:
        x, y, cost, direction = q.popleft()

        if cost >= dp[x][y][direction]:
            continue

        dp[x][y][direction] = cost

        if x == N - 1 and y == N - 1:
            min_cost = min(min_cost, cost)
            continue

        for i in range(4):
            mx, my = x + dx[i], y + dy[i]
            if 0 <= mx < N and 0 <= my < N and board[mx][my] == 0 and visited[mx][my]:
                visited[mx][my] = False

                if i > 1:
                    current_direction = 0
                else:
                    current_direction = 1

                if direction == current_direction:
                    q.append([mx, my, cost + 100, current_direction])
                else:
                    q.append([mx, my, cost + 600, current_direction])

                visited[mx][my] = True

    return min_cost