def solution(s):
    answer = ''
    s_len = len(s)
    if s_len % 2 == 0 :
        answer = s[s_len//2 - 1 : s_len//2 + 1]
    else :
        answer = s[s_len//2]
    return answer