import sys


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
M = int(sys.stdin.readline().strip())

parent = [i for i in range(N + 1)]

for i in range(1, N + 1) :
    data = list(map(int, (sys.stdin.readline()).split()))

    for j in range(N) :
        if data[j] == 1 :
            union_parent(i, j + 1, parent)

plan = list(map(int, (sys.stdin.readline()).split()))
standard = find_parent(plan[0], parent)
state = True

for i in range(1, len(plan)) :
    if standard != find_parent(plan[i], parent) :
        state = False
        break

if state :
    print("YES")
else :
    print("NO")
