N, L, R = map(int, input().split())

A, new_union, union, move_state, count = [], [], [], False, 0

for _ in range(N) :
    A.append(list(map(int, input().split())))


def move_country(s_row, s_colum):
    global union_people, union
    if s_row - 1 >= 0 and L <= abs(A[s_row][s_colum] - A[s_row - 1][s_colum]) <= R and not (s_row - 1, s_colum) in union :
        new_union.append((s_row - 1, s_colum))
        union.append((s_row - 1, s_colum))
        move_country(s_row - 1, s_colum)
    if s_colum - 1 >= 0 and L <= abs(A[s_row][s_colum] - A[s_row][s_colum - 1]) <= R and not (s_row, s_colum - 1) in union:
        new_union.append((s_row, s_colum - 1))
        union.append((s_row, s_colum - 1))
        move_country(s_row, s_colum - 1)
    if s_row + 1 < N and L <= abs(A[s_row][s_colum] - A[s_row + 1][s_colum]) <= R and not (s_row + 1, s_colum) in union:
        new_union.append((s_row + 1, s_colum))
        union.append((s_row + 1, s_colum))
        move_country(s_row + 1, s_colum)
    if s_colum + 1 < N and L <= abs(A[s_row][s_colum] - A[s_row][s_colum + 1]) <= R and not (s_row, s_colum + 1) in union:
        new_union.append((s_row, s_colum + 1))
        union.append((s_row, s_colum + 1))
        move_country(s_row, s_colum + 1)


while True:
    union, move_state = [], False
    for s_row in range(N) :
        for s_colum in range(N):
            if not (s_row, s_colum) in union:
                new_union, sum_people = [], 0

                union.append((s_row, s_colum))
                new_union.append((s_row, s_colum))

                move_country(s_row, s_colum)

                if len(new_union) >= 2:
                    move_state = True
                    for row, colum in new_union:
                        sum_people += A[row][colum]

                    distribution = sum_people // (len(new_union))

                    for row, colum in new_union:
                        A[row][colum] = distribution
                else:
                    union.remove((s_row, s_colum))

    if not move_state:
        break
    else :
        count += 1

print(count)