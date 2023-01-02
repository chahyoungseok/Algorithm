import sys

lines = sys.stdin.readlines()
for data in lines :
    data = list(map(float, (data.split())))
    points = []
    for j in range(4) :
        point = [data[j * 2], data[j * 2 + 1]]
        if point in points :
            standard_point = point
            points.remove(point)
        else :
            points.append(point)

    mid = [(points[0][0] + points[1][0]) / 2, (points[0][1] + points[1][1]) / 2]

    x = mid[0] * 2 - standard_point[0]
    y = mid[1] * 2 - standard_point[1]

    print(f"{x:0.3f} {y:0.3f}")