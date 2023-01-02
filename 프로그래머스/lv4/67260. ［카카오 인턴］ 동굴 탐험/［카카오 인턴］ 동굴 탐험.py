from collections import deque


def solution(n, path, order):
    edges, degree = [[] for _ in range(n)], [True for _ in range(n)]
    orders = [[] for _ in range(n)]

    for A, B in path:
        edges[A].append(B)
        edges[B].append(A)

    for A, B in order:
        orders[A].append(B)
        degree[B] = False

        if B == 0 :
            return False

    visited = [True for _ in range(n)]
    keep = {}
    q = deque()
    q.append(0)
    while q:
        node = q.popleft()
        visited[node] = False

        for i in orders[node]:
            degree[i] = True
            if i in keep.keys() and keep[i] :
                keep[i] = False
                q.append(i)

        for e in edges[node]:
            if visited[e] :
                if degree[e] :
                    q.append(e)
                else :
                    keep[e] = True

    for i in range(n) :
        if not degree[i] :
            return False
    return True