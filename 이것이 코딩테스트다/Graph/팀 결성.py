def find_parent(parent, x) :
    if parent[x] != x :
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
parent = [0] * (N + 1)

for i in range(1, N) :
    parent[i] = i

for _ in range(M) :
    w, a, b = map(int, input().split())
    if w == 0 :
        union_parent(parent, a, b)
    else :
        if find_parent(parent, a) == find_parent(parent, b) :
            print("Yes")
        else :
            print("No")