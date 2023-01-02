import copy


def find_parent(parent, x) :
    if parent[x] != x :
        return find_parent(parent, parent[x])
    return parent[x]


def union_parent(parent, a, b) :
    a = find_parent(parent, a)
    b = find_parent(parent, b)

    if a > b :
        parent[a] = b
    else:
        parent[b] = a


def dfs(board, x, y, visited, arr) :
    dx = [1, 0, -1, 0]
    dy = [0, 1, 0, -1]

    visited[x][y] = 0
    arr.append([x,y])

    for i in range(4) :
        mx, my = x + dx[i], y + dy[i]
        if mx >= 0 and mx < len(board) and my >= 0 and my < len(board[0]) :
            if board[mx][my] == 1 and visited[mx][my] == 1 :
                dfs(board, mx, my, visited, arr)
    return arr


def check_between(islands, island1, island2, point_1, point_2) :
    if point_1[0] == point_2[0] :
        if point_2[1] > point_1[1] :
            b, s = point_2[1], point_1[1]
        else :
            s, b = point_2[1], point_1[1]

        for i in range(s + 1, b) :
            for island in islands :
                if [point_1[0], i] in island :
                    return False
    elif point_1[1] == point_2[1] :
        if point_2[0] > point_1[0] :
            b, s = point_2[0], point_1[0]
        else :
            s, b = point_2[0], point_1[0]

        for i in range(s + 1, b) :
            for island in islands:
                if [i, point_1[1]] in island :
                    return False
    return True


def distances(islands, island1, island2) :
    min_distance = int(1e9)
    for x_1, y_1 in island1 :
        for x_2, y_2 in island2 :
            if x_1 == x_2 :
                d = abs(y_2 - y_1) - 1
                if d >= 2 and check_between(islands, island1, island2, [x_1, y_1], [x_2, y_2]) :
                    min_distance = min(min_distance, d)
            elif y_1 == y_2 :
                d = abs(x_2 - x_1) - 1
                if d >= 2 and check_between(islands, island1, island2, [x_1, y_1], [x_2, y_2]) :
                    min_distance = min(min_distance, d)
    if min_distance == int(1e9) :
        return -1
    return min_distance


N, M = map(int, input().split())

board, visited, islands = [], [], []
for _ in range(N) :
    board.append(list(map(int, input().split())))

visited = copy.deepcopy(board)
for i in range(N) :
    for j in range(M) :
        if visited[i][j] == 1 :
            islands.append(dfs(board, i, j, visited, []))

island_len, island_distance = len(islands), []
for i in range(island_len) :
    for j in range(island_len) :
        if i == j :
            continue
        island_distance.append([distances(islands, islands[i], islands[j]), i + 1, j + 1])

island_distance = sorted(island_distance, key=lambda x : x[0])
total, parent = 0, [0] * (island_len + 1)
for i in range(1, island_len + 1) :
    parent[i] = i

for cost, a, b in island_distance :
    if find_parent(parent, a) != find_parent(parent, b) and cost >= 2 :
        total += cost
        union_parent(parent, a, b)

standard, parent_state = find_parent(parent, 1), True
for i in range(2, island_len + 1) :
    if standard != find_parent(parent, i) :
        parent_state = False
        break

if not parent_state :
    print(-1)
else :
    print(total)