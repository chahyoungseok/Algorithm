min_distance = int(1e9)

def isMove(word_1, word_2) :
    index = 0
    for i in range(len(word_1)) :
        if word_1[i] != word_2[i] :
            index += 1
            
    if index == 1 :
        return True
    else :
        return False

def bfs(current, target, words, distance, visited) :
    global min_distance
    
    if current == target :
        min_distance = min(min_distance, distance)
        return 
    
    for i in range(len(words)) :
        if isMove(current, words[i]) and not visited[i]:
            visited[i] = True
            bfs(words[i], target, words, distance + 1, visited)
        
    
def solution(begin, target, words):
    visited = [False] * len(words)
    bfs(begin, target, words, 0, visited)
    
    if min_distance == int(1e9) :
        return 0
    else :
        return min_distance