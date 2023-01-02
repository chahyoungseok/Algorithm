def solution(enroll, referral, seller, amount):
    enroll_len = len(enroll)
    answer = []
    graph = {}
    
    for i in range(enroll_len) :
        graph[enroll[i]] = [referral[i], 0]
    
    for i in range(len(seller)) : 
        pure = amount[i] * 100
        who = seller[i]
        index = 0
        while True :
            index += 1
            t = int(pure * 0.1)
            graph[who][1] += pure - t
            pure = t
            who = graph[who][0]
            if who == "-" or pure == 0:
                break
            
            
    for i in range(enroll_len) :
        answer.append(graph[enroll[i]][1])
    
    return answer