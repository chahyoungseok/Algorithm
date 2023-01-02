def find_parent(parent, x) :
    if parent[x] != x:
        parent[x] = find_parent(parent, parent[x])
    return parent[x]

def union_parent(parent, a, b) :
    a = find_parent(parent, a)
    b = find_parent(parent, b)
    if a > b :
        parent[a] = b
    else :
        parent[b] = a


N, M = map(int, input().split())

parent, edges, cost_sum, origin_cost = [0] * N, [], 0, 0

for i in range(N) :
    parent[i] = i

for _ in range(M) :
    X, Y, Z = map(int, input().split())
    origin_cost += Z
    edges.append((Z, X, Y))

edges.sort()

for edge in edges :
    if find_parent(parent, edge[1]) != find_parent(parent, edge[2]) :
        cost_sum += edge[0]
        union_parent(parent, edge[1], edge[2])

print(origin_cost - cost_sum)