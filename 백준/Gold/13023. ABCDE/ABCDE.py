import sys


N, M = map(int, (sys.stdin.readline()).split())

edges = [[] for _ in range(N)]
for _ in range(M) :
    a, b = map(int, ((sys.stdin.readline()).split()))
    edges[a].append(b)
    edges[b].append(a)


def dfs(current, visited, dist) :
    if dist == 4 :
        return True

    for j in edges[current] :
        if visited[j] :
            visited[j] = False
            if dfs(j, visited, dist + 1) :
                return True
            visited[j] = True

    return False


visited, state = [True for _ in range(N)], False
for i in range(N) :
    visited[i] = False
    if dfs(i, visited, 0) :
        state = True
        break
    visited[i] = True

if state :
    print(1)
else :
    print(0)