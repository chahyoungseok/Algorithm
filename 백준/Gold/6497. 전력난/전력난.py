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


while True:
    m, n = map(int, sys.stdin.readline().strip().split())
    if m == 0 and n == 0 :
        break

    parent = [x for x in range(m + 1)]

    edges = []
    total_weight = 0
    for _ in range(n):
        x, y, z = map(int, sys.stdin.readline().strip().split())
        total_weight += z
        heapq.heappush(edges, [z, x, y])

    count, result = 0, 0
    while edges and count != m - 1:
        weight, a, b = heapq.heappop(edges)

        if find_parent(a) == find_parent(b):
            continue

        union_parent(a, b)
        count += 1
        result += weight

    print(total_weight - result)

