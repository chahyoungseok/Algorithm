def solution(s):
    answer = []
    s = (s[2:-2]).split("},{")
    s = sorted(s, key=lambda x : len(x))

    for i in s :
        k = i.split(',')
        for j in k :
            if int(j) not in answer :
                answer.append(int(j))
                break
    return answer