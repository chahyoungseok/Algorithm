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

parent, data, state = [0] * N, [], True

for i in range(N) :
    parent[i] = i

for i in range(N) :
    info = list(map(int, input().split()))
    for j in range(N) :
        if info[j] == 1 and not [j,i] in data :
            data.append([i,j])
            union_parent(parent, i, j)

plans = list(map(int, input().split()))

for i in range(N - 1) :
    if find_parent(parent, i) != find_parent(parent, i + 1) :
        state = False

if state :
    print("YES")
else :
    print("NO")