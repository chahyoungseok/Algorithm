def solution(lottos, win_nums):
    answer = []
    total, gap = 0, 0
    for i in lottos :
        if i == 0 :
            gap += 1
            continue
        
        for j in win_nums :
            if i == j :
                total += 1
                break
    
    max_grade = total + gap
    
    if max_grade == 0 or max_grade == 1 :
        answer.append(6)
    else :
        answer.append(7 - max_grade)
    
    if total == 0 or total == 1 :
        answer.append(6)
    else :
        answer.append(7 - total)
        
    return answer