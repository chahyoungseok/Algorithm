import sys


def swap(p1, p2) :
    temp = board[p1[0]][p1[1]]
    board[p1[0]][p1[1]] = board[p2[0]][p2[1]]
    board[p2[0]][p2[1]] = temp


def check() :
    max_total = 0
    for x in range(N) :
        standard, total = board[x][0], 1
        for y in range(1, N) :
            if standard == board[x][y] :
                total += 1
            else :
                max_total = max(max_total, total)
                standard, total = board[x][y], 1
        max_total = max(max_total, total)
    for y in range(N) :
        standard, total = board[0][y], 1
        for x in range(1, N) :
            if standard == board[x][y] :
                total += 1
            else :
                max_total = max(max_total, total)
                standard, total = board[x][y], 1
        max_total = max(max_total, total)
    return max_total


N = int(input())
board, max_candy = [], 1
for _ in range(N) :
    board.append(list(sys.stdin.readline().strip()))

for i in range(N) :
    for j in range(N) :
        if i + 1 < N :
            swap([i, j], [i + 1, j])
            max_candy = max(check(), max_candy)
            swap([i, j], [i + 1, j])
        if j + 1 < N :
            swap([i, j], [i, j + 1])
            max_candy = max(check(), max_candy)
            swap([i, j], [i, j + 1])

print(max_candy)