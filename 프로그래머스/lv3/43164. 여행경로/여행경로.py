from collections import defaultdict


def dfs(start, r, ddict, target_len, visited):
    r.append(start)
    if len(r) == target_len:
        return r
    
    for i in range(len(ddict[start])):
        if visited[start][i] :
            visited[start][i] = False
            if dfs(ddict[start][i], r, ddict, target_len, visited) :
                return r
            visited[start][i] = True
        
    r.pop()
    return False
    
def solution(tickets):
    ddict, visited = defaultdict(list), defaultdict(list)
    r, target_len = [], len(tickets) + 1

    for ticket in tickets:
        ddict[ticket[0]].append(ticket[1])
        visited[ticket[0]].append(True)

    for key in ddict.keys():
        ddict[key] = sorted(ddict[key])

    return dfs("ICN", r, ddict, target_len, visited)