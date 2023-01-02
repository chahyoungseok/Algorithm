import sys
from collections import deque

N = int(input())
M = int(input())

edges = [[] for _ in range(N + 1)]
for _ in range(M) :
    a, b = map(int, (sys.stdin.readline()).split())
    edges[a].append(b)
    edges[b].append(a)

q = deque()
q.append(1)
visited = [True for _ in range(N + 1)]
visited[1], total = False, 0

while q :
    current = q.popleft()

    for i in edges[current] :
        if visited[i] :
            q.append(i)
            visited[i] = False
            total += 1

print(total)