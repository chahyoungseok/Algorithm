def solution(numbers, hand):
    result = ""
    left, right = [3, 0], [3, 2]

    for number in numbers:
        if number == 0:
            x, y = 3, 1
        else:
            x, y = (number - 1) // 3, (number - 1) % 3

        if y == 0:
            result += "L"
            left = [x, y]
        elif y == 2:
            result += "R"
            right = [x, y]
        else:
            left_distance = abs(left[0] - x) + abs(left[1] - y)
            right_distance = abs(right[0] - x) + abs(right[1] - y)
            if right_distance > left_distance:
                result += "L"
                left = [x, y]
            elif left_distance > right_distance:
                result += "R"
                right = [x, y]
            else:
                if hand == "left":
                    result += "L"
                    left = [x, y]
                else:
                    result += "R"
                    right = [x, y]

    return result