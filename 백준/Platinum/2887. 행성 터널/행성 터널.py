import sys, heapq


def find_parent(x, parent) :
    if parent[x] != x :
        return find_parent(parent[x], parent)
    return x


def union_parent(a, b, parent) :
    a = find_parent(a, parent)
    b = find_parent(b, parent)

    if a > b :
        parent[a] = b
    else :
        parent[b] = a


N = int(sys.stdin.readline().strip())

edges = []
storage = []
parent = [i for i in range(N + 1)]
for i in range(1, N + 1) :
    a, b, c = map(int, (sys.stdin.readline()).split())
    storage.append([a, b, c, i])

storage = sorted(storage, key=lambda x : x[0])
for i in range(N - 1) :
    heapq.heappush(edges, [abs(storage[i + 1][0] - storage[i][0]), storage[i + 1][3], storage[i][3]])

storage = sorted(storage, key=lambda x : x[1])
for i in range(N - 1) :
    heapq.heappush(edges, [abs(storage[i + 1][1] - storage[i][1]), storage[i + 1][3], storage[i][3]])

storage = sorted(storage, key=lambda x : x[2])
for i in range(N - 1) :
    heapq.heappush(edges, [abs(storage[i + 1][2] - storage[i][2]), storage[i + 1][3], storage[i][3]])

result, count = 0, 0
while count != N - 1 :
    value, node_1, node_2 = heapq.heappop(edges)
    if find_parent(node_1, parent) != find_parent(node_2, parent) :
        union_parent(node_1, node_2, parent)
        result += value
        count += 1

print(result)