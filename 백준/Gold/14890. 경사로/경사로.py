import sys

N, L = map(int, (sys.stdin.readline()).split())
board, count = [], 0

for _ in range(N) :
    board.append(list(map(int, (sys.stdin.readline()).split())))


def check(type, index):
    # 가로줄 탐색
    if type == 0:
        current_height, same_height, state, j = board[index][0], 1, True, 1
        while N > j and state:
            if current_height == board[index][j]:
                same_height += 1
            elif current_height + 1 == board[index][j] and same_height >= L :
                current_height, same_height = board[index][j], 1
            elif current_height - 1 == board[index][j] and N - j >= L:
                for k in range(j + 1, j + L) :
                    if current_height - 1 != board[index][k] :
                        state = False
                        break
                if state :
                    current_height, same_height = board[index][j], 0
                    j += (L - 1)
            else :
                state = False
            j += 1

        if not state :
            current_height, same_height, state, j = board[index][N - 1], 1, True, N - 2
            while j >= 0 and state:
                if current_height == board[index][j]:
                    same_height += 1
                elif current_height + 1 == board[index][j] and same_height >= L:
                    current_height, same_height = board[index][j], 1
                elif current_height - 1 == board[index][j] and j >= L - 1:
                    for k in range(j - 1, j - L - 1, -1):
                        if current_height - 1 != board[index][k]:
                            return False
                    if state:
                        current_height, same_height = board[index][j], 0
                        j -= (L - 1)
                else:
                    return False
                j -= 1
        return True
    else :
        current_height, same_height, state, j = board[0][index], 1, True, 1
        while N > j and state:
            if current_height == board[j][index]:
                same_height += 1
            elif current_height + 1 == board[j][index] and same_height >= L:
                current_height, same_height = board[j][index], 1
            elif current_height - 1 == board[j][index] and N - j >= L:
                for k in range(j + 1, j + L):
                    if current_height - 1 != board[k][index]:
                        state = False
                        break
                if state:
                    current_height, same_height = board[j][index], 0
                    j += (L - 1)
            else:
                state = False
            j += 1

        if not state:
            current_height, same_height, state, j = board[N - 1][index], 1, True, N - 2
            while j >= 0 and state:
                if current_height == board[j][index]:
                    same_height += 1
                elif current_height + 1 == board[j][index] and same_height >= L:
                    current_height, same_height = board[j][index], 1
                elif current_height - 1 == board[j][index] and j >= L - 1:
                    for k in range(j - 1, j - L - 1, -1):
                        if current_height - 1 != board[k][index]:
                            return False
                    if state:
                        current_height, same_height = board[j][index], 0
                        j -= (L - 1)
                else:
                    return False
                j -= 1
        return True


for i in range(N) :
    if check(0, i) :
        count += 1
    if check(1, i) :
        count += 1

print(count)
