import heapq, sys


def find_parent(x):
    if parent[x] != x:
        parent[x] = find_parent(parent[x])
    return parent[x]


def union_parent(a, b):
    if a == b:
        return

    a = find_parent(a)
    b = find_parent(b)

    if a > b:
        parent[a] = b
    else:
        parent[b] = a


N, M = map(int, sys.stdin.readline().strip().split())

parent = [x for x in range(N + 1)]

edges = []
for _ in range(M):
    A, B, C = map(int, sys.stdin.readline().strip().split())
    heapq.heappush(edges, [C, A, B])


count, max_value, result = 0, 0, 0
while edges and count != N - 1:
    weight, a, b = heapq.heappop(edges)

    if find_parent(a) == find_parent(b):
        continue

    union_parent(a, b)
    count += 1
    max_value = max(max_value, weight)
    result += weight

print(result - max_value)