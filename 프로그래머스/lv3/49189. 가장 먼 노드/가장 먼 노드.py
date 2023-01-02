from collections import deque

def solution(n, edge) :
    graph = [[] for _ in range(n + 1)]
    visited = [False] * (n + 1)

    for edge_case in edge :
        graph[edge_case[0]].append(edge_case[1])
        graph[edge_case[1]].append(edge_case[0])

    queue = deque()
    queue.append([1, 0])
    answer, max_dist = 0, 0

    visited[1] = True
    while queue:
        v, dist = queue.popleft()

        if dist > max_dist:
            max_dist = dist
            answer = 1
        elif dist == max_dist:
            answer += 1

        for i in graph[v]:
            if not visited[i]:
                queue.append([i, dist + 1])
                visited[i] = True

    return answer