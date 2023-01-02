def solution(msg):
    answer, alpha_dict, msg_len = [], {}, len(msg)
    for i in range(ord('A'), ord('Z') + 1):
        alpha_dict[chr(i)] = i + 1 - ord('A')

    i = 0
    while msg_len > i:
        size, state = 1, True
        target = msg[i: i + size]

        while target in alpha_dict.keys():
            size += 1
            if i + size > msg_len :
                state = False
                break
            target = msg[i : i + size]

        i += size - 1
        if state :
            answer.append(alpha_dict[target[: -1]])
        else :
            answer.append(alpha_dict[target])

        alpha_dict[target] = len(alpha_dict.keys()) + 1
    return answer