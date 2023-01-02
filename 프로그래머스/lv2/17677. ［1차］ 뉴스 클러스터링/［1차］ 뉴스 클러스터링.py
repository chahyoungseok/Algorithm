def solution(str1, str2):
    str1, str2 = str1.lower(), str2.lower()
    dict_1, dict_2 = {}, {}
    inter_jacad, union_jacad = 0, 0
    alp = "abcdefghijklmnopqrstuvwxyz"

    for i in range(len(str1) - 1):
        target = str1[i: i + 2]
        if target[0] not in alp or target[1] not in alp :
            continue
        if target in dict_1.keys():
            dict_1[target] += 1
        else:
            dict_1[target] = 1

    for i in range(len(str2) - 1):
        target = str2[i: i + 2]
        if target[0] not in alp or target[1] not in alp :
            continue
        if target in dict_2.keys():
            dict_2[target] += 1
        else:
            dict_2[target] = 1

    for i in dict_1.keys():
        union_jacad += dict_1[i]
        for j in dict_2.keys():
            if i == j:
                value = min(dict_1[i], dict_2[i])
                inter_jacad += value
                union_jacad -= value

    for i in dict_2.keys() :
        union_jacad += dict_2[i]

    if union_jacad == 0 :
        return 65536

    return int(inter_jacad / union_jacad * 65536)