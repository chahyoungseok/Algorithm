import sys
from itertools import combinations


def cal_distance(p1, p2) :
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


N, M = map(int, (sys.stdin.readline()).split())
board, chi, house, min_distance = [], [], [], int(1e9)

for i in range(N) :
    data = list(map(int, (sys.stdin.readline()).split()))
    for j in range(N) :
        if data[j] == 2 :
            chi.append([i, j])
        elif data[j] == 1 :
            house.append([i, j])
    board.append(data)

for combin in combinations(chi, M) :
    total_dist = 0
    for h1 in house :
        house_chi = int(1e9)
        for c1 in combin :
            house_chi = min(house_chi, cal_distance(h1, c1))
        total_dist += house_chi

    min_distance = min(min_distance, total_dist)
print(min_distance)
