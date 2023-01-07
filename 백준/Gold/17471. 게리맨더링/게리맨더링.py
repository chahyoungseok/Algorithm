import sys
from itertools import combinations
from collections import deque


def check_graph(elements) :
    q = deque()
    q.append(elements[0])
    visited = [True for _ in range(N + 1)]
    visited[elements[0]] = False
    size_sum = size[elements[0] - 1]

    while q :
        current_node = q.popleft()

        for node in graph[current_node] :
            if visited[node] and node in elements :
                visited[node] = False
                size_sum += size[node - 1]
                q.append(node)

    for element in elements :
        if visited[element] :
            return -1, False

    return size_sum, True


N = int(sys.stdin.readline().strip())
size = list(map(int, (sys.stdin.readline()).split()))

graph = [[] for _ in range(N + 1)]

for i in range(1, N + 1) :
    data = list(map(int, (sys.stdin.readline()).split()))
    graph[i] = data[1:]

min_result = int(1e9)
for i in range(1, N // 2 + 1) :
    for combi in combinations(range(1, N + 1), i) :
        non_combi = []
        for j in range(1, N + 1) :
            if j not in combi :
                non_combi.append(j)

        sum_1, state_1 = check_graph(combi)
        sum_2, state_2 = check_graph(non_combi)

        if state_1 and state_2 :
            min_result = min(min_result, abs(sum_1 - sum_2))

if min_result == int(1e9) :
    print(-1)
else :
    print(min_result)