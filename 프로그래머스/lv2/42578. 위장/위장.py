def solution(clothes):
    answer = 1
    cloth_combin = {}

    for cloth in clothes :
        if cloth[1] in cloth_combin.keys() :
            cloth_combin[cloth[1]] += 1
        else :
            cloth_combin[cloth[1]] = 1

    for num in cloth_combin.values() :
        answer *= (num + 1)

    return answer - 1