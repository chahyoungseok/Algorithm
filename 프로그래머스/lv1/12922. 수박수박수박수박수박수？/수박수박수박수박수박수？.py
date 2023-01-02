def solution(n):
    str1, str2 = "수", "박"
    answer = ''
    for i in range(n) :
        if i % 2 == 0 :
            answer += str1
        else :
            answer += str2

    return answer