from itertools import combinations

N, M = map(int, input().split())
restaurant, house, min_sum = [], [], int(1e9)

for i in range(N) :
    city = list(map(int, input().split()))
    for j in range(N) :
        if city[j] == 1 :
            house.append((i,j))
        elif city[j] == 2 :
            restaurant.append((i, j))

combin = list(combinations(restaurant,M))

for restaurants in combin :
    sum_d = 0

    for hx, hy in house :
        min_d = int(1e9)
        for rx, ry in restaurants:
            min_d = min(min_d, abs(hx - rx) + abs(hy - ry))
        sum_d += min_d

    min_sum = min(sum_d,min_sum)

print(min_sum)