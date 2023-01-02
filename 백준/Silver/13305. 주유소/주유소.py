city_len = int(input())
city_between = list(map(int, input().split()))
city_oil = list(map(int, input().split()))

oil_sum = 0

record_oil = [[city_oil[city_len - 2], city_between[city_len - 2]]] #비용, 거리
for i in range(city_len - 3, -1, -1) :
    while record_oil and record_oil[len(record_oil) - 1][0] >= city_oil[i] :
        city_between[i] += record_oil[len(record_oil) - 1][1]
        record_oil.pop(len(record_oil) - 1)

    if not record_oil :
        record_oil = [[city_oil[i], city_between[i]]]
    else :
        record_oil.append([city_oil[i], city_between[i]])


for money, distance in record_oil :
    oil_sum += money * distance

print(oil_sum)