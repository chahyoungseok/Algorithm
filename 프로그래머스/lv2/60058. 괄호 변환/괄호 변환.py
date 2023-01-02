def is_balanced(p) :
    count = 0
    for i in range(len(p)) :
        if p[i] == '(' :
            count += 1
        else :
            count -= 1
        if count == 0 :
            return i

def is_perfected(p) :
    count = 0
    for i in range(len(p)) :
        if p[i] == '(' :
            count += 1
        else :
            count -= 1
        if count < 0 :
            return False
    return True

def solution(p):
    answer = ""
    if p == "" :
        return ""

    balance_index = is_balanced(p)
    u = p[:balance_index + 1]
    v = p[balance_index + 1:]

    if is_perfected(u) :
        answer = u + solution(v)
    else :
        answer = '('
        answer += solution(v)
        answer += ')'
        u = list(u[1:-1])
        for i in range(len(u)) :
            if u[i] == '(' :
                u[i] = ')'
            else :
                u[i] = '('
        answer += "".join(u)
    return answer