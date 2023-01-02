N, C = map(int, input().split())
house_loc, result = [], 0
for _ in range(N) :
    house_loc.append(int(input()))

house_loc.sort()
start = 1
end = house_loc[-1] - house_loc[0]

while(start <= end) :
    mid = (start + end) // 2
    pre, setRouter = house_loc[0], 1

    for i in range(1, N) :
        if house_loc[i] - pre >= mid :
            pre = house_loc[i]
            setRouter += 1

    if setRouter >= C :
        start = mid + 1
        result = mid
    else :
        end = mid - 1

print(result)