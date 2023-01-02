T = int(input())

for _ in range(T) :
    x, y = map(int, input().split())
    cur_location, cur_speed, total_distance = 0, 0, y - x
    result = 0

    if y - x == 1 :
        print(1)
    else :
        while total_distance >= (cur_speed * (cur_speed + 1)) // 2 + (cur_speed * (cur_speed - 1)) // 2 :
            cur_speed += 1

        cur_speed -= 1
        result = (cur_speed - 1) * 2 + 1
        cur_location = total_distance - (cur_speed * (cur_speed + 1)) // 2 - (cur_speed * (cur_speed - 1)) // 2

        for i in range(cur_speed, 0, -1) :
            while cur_location >= i :
                cur_location -= i
                result += 1

        print(result)