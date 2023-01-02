def distance(point1, point2) :
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


T = int(input())

for _ in range(T) :
    x1, y1, x2, y2 = map(int, input().split())
    total = 0
    n = int(input())
    for i in range(n) :
        cx, cy, r = map(int, input().split())

        if (distance([cx, cy], [x1, y1]) > r and distance([cx, cy], [x2, y2]) < r) or (distance([cx, cy], [x1, y1]) < r and distance([cx, cy], [x2, y2]) > r) :
            total += 1

    print(total)


