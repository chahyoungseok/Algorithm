def solution(n, build_frame):
    answer = []

    def check(board):
        for x, y, a in board:
            # 기둥
            if a == 0:
                if y == 0 or ([x, y - 1, 0] in board) or ([x, y, 1] in board) or ([x - 1, y, 1] in board):
                    continue
                return False
            # 보
            else:
                if ([x + 1, y - 1, 0] in board) or ([x, y - 1, 0] in board) or ([x - 1, y, 1] in board and [x + 1, y, 1] in board):
                    continue
                return False
        return True

    for x, y, a, b in build_frame:
        if b == 1:
            answer.append([x, y, a])
            if not check(answer):
                answer.remove([x, y, a])
        else:
            answer.remove([x, y, a])
            if not check(answer):
                answer.append([x, y, a])

    answer = sorted(answer, key=lambda x: (x[0], +x[1], +x[2]))
    return answer