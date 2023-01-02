def solution(survey, choices):
    answer = ""
    mbti_case = [["R", "T"], ["C", "F"], ["J", "M"], ["A", "N"]]
    mbti = { "R" : 0, "T" : 0, "C" : 0, "F" : 0, "J" : 0, "M" : 0, "A" : 0, "N" : 0 }
    
    for i in range(len(survey)) :
        target, point = survey[i], choices[i]
        
        if point > 4 :
            mbti[target[1]] += point - 4
        else :
            mbti[target[0]] += 4 - point
    
    for i in range(4) :
        mbti_target = mbti_case[i]
        
        if mbti[mbti_target[0]] >= mbti[mbti_target[1]] :
            answer += mbti_target[0]
        else :
            answer += mbti_target[1]
        
    return answer