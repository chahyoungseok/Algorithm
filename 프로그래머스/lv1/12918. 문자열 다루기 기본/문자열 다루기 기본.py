def solution(s):
    answer = True
    for i in s :
        if ord(i) < 48 or ord(i) > 57 :
            answer = False
            break

    if not (len(s) == 4 or len(s) == 6) :
        answer = False
    return answer