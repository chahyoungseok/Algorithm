N, M = map(int, input().split())

board, min_count = [], int(1e9)

for _ in range(N) :
    board.append(input())

sample = []
for i in range(8):
    case = []
    for j in range(8):
        if (i + j) % 2 == 0 :
            case.append('B')
        else:
            case.append('W')
    sample.append(case)

for i in range(0, N - 7) :
    for j in range(0, M - 7) :
        count = 0
        for a in range(i, i + 8) :
            for b in range(j, j + 8) :
                if board[a][b] != sample[a - i][b - j] :
                    count += 1

        min_count = min(min_count, count, 64 - count)

print(min_count)