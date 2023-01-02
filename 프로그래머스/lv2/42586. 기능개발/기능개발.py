from collections import deque
def solution(progresses, speeds):
    progresses, speeds = deque(progresses), deque(speeds)
    answer = []
    
    while progresses :
        index = 0
        for i in range(0, len(progresses)) :
            progresses[i] = progresses[i] + speeds[i]
            
        while progresses and progresses[0] >= 100 :
            progresses.popleft()
            speeds.popleft()
            index+=1
            
        if index != 0 :
            answer.append(index)
    
    return answer