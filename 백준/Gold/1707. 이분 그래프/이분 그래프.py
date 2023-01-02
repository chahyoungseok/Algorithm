import sys
sys.setrecursionlimit(10 ** 6)


def dfs(start, visited) :
    for i in edges[start] :
        if color[i] == -1:
            if color[start] == 1:
                color[i] = 2
            else:
                color[i] = 1
        elif color[i] == color[start]:
            return False

        if visited[i] :
            visited[i] = False
            if not dfs(i, visited):
                return False

    return True


K = int(input())
for _ in range(K) :
    V, E = map(int, input().split())

    edges, state = [[] for _ in range(V + 1)], True
    color = [-1 for _ in range(V + 1)]
    for _ in range(E) :
        u, v = map(int, (sys.stdin.readline()).split())
        edges[u].append(v)
        edges[v].append(u)

    visited = [True for _ in range(V + 1)]

    for i in range(1, V + 1) :
        if color[i] == -1 :
            color[i], visited[i] = 1, False
            if not dfs(i, visited) :
                state = False

    if state :
        print("YES")
    else:
        print("NO")