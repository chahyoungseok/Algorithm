from collections import defaultdict


def solution(gems):
    jewelry_len, gems_len, dic_gems = len(set(gems)), len(gems), defaultdict(int)
    start, end, answer = 0, 0, [0, int(1e9)]

    dic_gems[gems[0]] += 1
    while start < gems_len and end < gems_len :
        if jewelry_len > len(dic_gems.keys()) :
            end += 1
            if end == gems_len :
                break
            dic_gems[gems[end]] += 1
        else :
            if answer[1] - answer[0] > end - start :
                answer = [start, end]
            if dic_gems[gems[start]] == 1 :
                del dic_gems[gems[start]]
            else :
                dic_gems[gems[start]] -= 1
            start += 1

    answer[0] += 1
    answer[1] += 1
    return answer