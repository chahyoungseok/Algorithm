from collections import deque

def solution(numbers, target) :
    result = deque([0])
    q = deque(numbers)
    answer = 0
    
    while q :
        cur = q.popleft()
        for _ in range(len(result)) :
            sel = result.popleft()
            result.append(sel + cur)
            result.append(sel - cur)
    
    for i in result :
        if i == target :
            answer += 1
    
    return answer