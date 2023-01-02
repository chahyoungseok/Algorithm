def solution(brown, yellow):
    answer = []

    w_h = brown // 2

    for i in range(1, w_h - 1) :
        weight = w_h - i
        height = w_h - weight + 2

        if yellow == (weight - 2) * (height - 2) :
            answer.append(weight)
            answer.append(height)
            break

    return answer