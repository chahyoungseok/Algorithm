def solution(number, k):
    sub, index, size = 0, 0, len(number)

    while sub != k and size - 1 > index:
        if number[index] != '9' and number[index] < number[index + 1] :
            number = number[:index] + number[index + 1:]
            # number.remove(number[index])
            sub += 1
            index -= 1
            size -= 1
        else :
            index += 1

        if index < 0 :
            index = 0

    return "".join(number[0:size - k + sub])