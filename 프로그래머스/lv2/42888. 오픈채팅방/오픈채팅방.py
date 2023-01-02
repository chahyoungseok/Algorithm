def solution(record):
    user = {}
    answers = []
    result = []
    for i in record :
        data = i.split()
        if data[0] == "Change" :
            user[data[1]] = data[2]
        else :
            if data[0] == "Enter" :
                user[data[1]] = data[2]
            answers.append([data[0], data[1]])    
        
    for answer in answers :
        if answer[0] == "Enter" :
            result.append(user[answer[1]]+"님이 들어왔습니다.")
        else :
            result.append(user[answer[1]]+"님이 나갔습니다.")
        
    
    
    return result