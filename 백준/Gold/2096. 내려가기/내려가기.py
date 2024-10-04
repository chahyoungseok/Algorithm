import sys

N = int(sys.stdin.readline().strip())
dp = [[[-sys.maxsize, sys.maxsize] for _ in range(3)] for _ in range(2)]

for i in range(3):
    dp[0][i] = [0, 0]

for i in range(N):
    board = list(map(int, sys.stdin.readline().strip().split()))

    dp[1][0] = [
        max(dp[1][0][0], dp[0][0][0] + board[0], dp[0][1][0] + board[0]),
        min(dp[1][0][1], dp[0][0][1] + board[0], dp[0][1][1] + board[0])
    ]

    dp[1][1] = [
        max(dp[1][1][0], dp[0][0][0] + board[1], dp[0][1][0] + board[1], dp[0][2][0] + board[1]),
        min(dp[1][1][1], dp[0][0][1] + board[1], dp[0][1][1] + board[1], dp[0][2][1] + board[1])
    ]

    dp[1][2] = [
        max(dp[1][2][0], dp[0][1][0] + board[2], dp[0][2][0] + board[2]),
        min(dp[1][2][1], dp[0][1][1] + board[2], dp[0][2][1] + board[2])
    ]

    for j in range(3):
        dp[0][j] = dp[1][j]
        dp[1][j] = [-sys.maxsize, sys.maxsize]

max_result, min_result = -sys.maxsize, sys.maxsize
for max_value, min_value in dp[0]:
    if max_value > max_result:
        max_result = max_value
    if min_value < min_result:
        min_result = min_value

print(str(max_result) + " " + str(min_result))