import sys

T = int(sys.stdin.readline())
dx, dy = [0, 1, 0, -1], [1, 0, -1, 0]
for _ in range(T) :
    order = list(sys.stdin.readline().strip())
    history, current_point, location = [[0, 0]], [0, 0], 0

    for i in order :
        if i == "L" :
            location = (location - 1) % 4
        elif i == "R" :
            location = (location + 1) % 4
        elif i == "F" :
            current_point = [current_point[0] + dx[location], current_point[1] + dy[location]]
            history.append(current_point)
        elif i == "B" :
            current_point = [current_point[0] - dx[location], current_point[1] - dy[location]]
            history.append(current_point)

    max_0, min_0 = 0, int(1e9)
    max_1, min_1 = 0, int(1e9)
    for a, b in history :
        if a > max_0 :
            max_0 = a
        if min_0 > a :
            min_0 = a
        if b > max_1:
            max_1 = b
        if min_1 > b:
            min_1 = b
    print((max_0 - min_0) * (max_1 - min_1))
