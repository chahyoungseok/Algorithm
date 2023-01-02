import sys, heapq


def find_parent(parent, x) :
    if parent[x] != x :
        return find_parent(parent, parent[x])
    return x


def union_parent(parent, x, y) :
    x = find_parent(parent, x)
    y = find_parent(parent, y)

    if x > y :
        parent[x] = y
    else :
        parent[y] = x


N = int(sys.stdin.readline().strip())
M = int(sys.stdin.readline().strip())

q, parent = [], [i for i in range(N + 1)]

for _ in range(M) :
    a, b, c = map(int, (sys.stdin.readline()).split())
    if a != b :
        heapq.heappush(q, [c, a, b])

total = 0
while q :
    cost, a, b = heapq.heappop(q)
    if find_parent(parent, a) != find_parent(parent, b) :
        union_parent(parent, a, b)
        total += cost

print(total)