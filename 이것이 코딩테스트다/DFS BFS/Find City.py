from collections import deque

N, M, K, X = map(int, input().split())

graph = [[] for _ in range(N + 1)]
visited = [False] * (N + 1)

for _ in range(M) :
    A, B = map(int, input().split())
    graph[A].append(B)

dist_to_node = []
queue = deque()
queue.append((X,0))
visited[X] = True

while queue :
    n, dist = queue.popleft()
    dist_to_node.append((n, dist))
    for i in graph[n] :
        if not visited[i] :
            queue.append((i, dist + 1))
            visited[i] = True

for i in range(N) :
    if dist_to_node[i][1] == K :
        visited[K] = False
        print(i + 1, end=' ')

if visited[K] :
    print(-1)