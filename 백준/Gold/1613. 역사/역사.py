import sys
from collections import deque

n, k = map(int, (sys.stdin.readline()).split())
edges = [[] for _ in range(n + 1)]
pre_events = [set() for _ in range(n + 1)]

for _ in range(k) :
    a, b = map(int, (sys.stdin.readline()).split())
    edges[b].append(a)

for i in range(1, n + 1) :
    q = deque(edges[i])

    for j in edges[i] :
        pre_events[i].add(j)

    visited = [True for _ in range(n + 1)]
    visited[i] = False

    while q :
        node = q.popleft()

        for e in edges[node] :
            if visited[e] :
                visited[e] = False
                pre_events[i].add(e)
                if e not in q :
                    q.append(e)

s = int(sys.stdin.readline().strip())
for _ in range(s) :
    a, b = map(int, (sys.stdin.readline()).split())

    if b in pre_events[a] :
        print(1)
    elif a in pre_events[b] :
        print(-1)
    else :
        print(0)