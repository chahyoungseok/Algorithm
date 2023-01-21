import sys
from collections import deque

N = int(sys.stdin.readline().strip())

q = deque()
times = [0 for _ in range(N + 1)]
degrees = [0 for _ in range(N + 1)]
reverse_edges = [[] for _ in range(N + 1)]
results = [0 for _ in range(N + 1)]

for i in range(1, N + 1) :
    data = list(map(int, (sys.stdin.readline()).split()))

    times[i] = data[0]
    data_len = len(data)

    if data_len == 2 :
        q.append(i)
        results[i] = data[0]
    else :
        degrees[i] += data_len - 2
        for j in data[1 : -1] :
            reverse_edges[j].append(i)

while q :
    node = q.popleft()

    for i in reverse_edges[node] :
        degrees[i] -= 1
        results[i] = max(results[i], results[node] + times[i])

        if degrees[i] == 0 :
            degrees[i] -= 1
            q.append(i)

for i in range(1, N + 1) :
    print(results[i])