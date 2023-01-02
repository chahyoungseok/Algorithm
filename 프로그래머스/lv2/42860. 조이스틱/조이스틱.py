def solution(name):
    answer, name_len = 0, len(name)
    n1, n2 = 0, 0

    for i in range(name_len):
        answer += min(ord(name[i]) - ord("A"), ord("Z") - ord(name[i]) + 1)

    for i in range(1, name_len) :
        if name[i] == "A" :
            n1 += 1
        else :
            break

    for i in range(name_len - 1, 0 , -1) :
        if name[i] == "A" :
            n2 += 1
        else :
            break

    move = name_len - min(n1, n2) - 1

    for i in range(name_len) :
        index = i + 1
        while name_len > index and name[index] == "A" :
            index += 1

        move = min(move, i * 2 + name_len - index, i + 2 * (name_len - index))

    return answer + move