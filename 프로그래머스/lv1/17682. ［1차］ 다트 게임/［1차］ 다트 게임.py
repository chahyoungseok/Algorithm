def solution(dartResult):
    bonus, total = [], []
    dartResult_len = len(dartResult)

    for i in range(dartResult_len):
        if dartResult[i] in "SDT":
            value = 0
            if bonus:
                value = int(dartResult[bonus[len(bonus) - 1][1] + 1:i])
            else:
                value = int(dartResult[:i])

            if dartResult[i] == "S":
                if dartResult_len > i + 1 and dartResult[i + 1] in "*#":
                    bonus.append([1, i + 1, dartResult[i + 1], value])
                else:
                    bonus.append([1, i, "n", value])
            if dartResult[i] == "D":
                if dartResult_len > i + 1 and dartResult[i + 1] in "*#":
                    bonus.append([2, i + 1, dartResult[i + 1], value])
                else:
                    bonus.append([2, i, "n", value])
            if dartResult[i] == "T":
                if dartResult_len > i + 1 and dartResult[i + 1] in "*#":
                    bonus.append([3, i + 1, dartResult[i + 1], value])
                else:
                    bonus.append([3, i, "n", value])
                    
    for b, index, p, value in bonus :
        total.append(value ** b)
        total_len = len(total)
        if p == "*" :
            total[total_len - 1] *= 2
            if total_len >= 2 : 
                total[total_len - 2] *= 2
        elif p == "#" :
            total[total_len - 1] *= -1
    
    return sum(total)