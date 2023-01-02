# 26
import sys
from collections import deque


N, M = map(int, (sys.stdin.readline()).split())

de, edges = [0 for _ in range(N + 1)], [[] for _ in range(N + 1)]
q, result = deque(), []

for _ in range(M) :
    a, b = map(int, (sys.stdin.readline()).split())
    edges[b].append(a)
    de[a] += 1


for i in range(1, N + 1) :
    if de[i] == 0 :
        q.append(i)

while q :
    current = q.popleft()
    result.append(current)

    for i in edges[current] :
        de[i] -= 1
        if de[i] == 0 :
            q.append(i)


for i in range(N - 1, -1, -1) :
    print(result[i], end=" ")