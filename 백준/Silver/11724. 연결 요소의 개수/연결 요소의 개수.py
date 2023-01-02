import sys


def find_parent(parent, x) :
    if parent[x] != x :
        return find_parent(parent, parent[x])
    return x


def union_parent(parent, a, b) :
    a = find_parent(parent, a)
    b = find_parent(parent, b)

    if a > b :
        parent[a] = b
    else :
        parent[b] = a


N, M = map(int, (sys.stdin.readline()).split())

parent = [0] + [(i + 1) for i in range(N)]
for _ in range(M) :
    u, v = map(int, (sys.stdin.readline()).split())
    union_parent(parent, u, v)

dic = {}
for i in range(1, N + 1) :
    target = find_parent(parent, i)
    if target not in dic.keys() :
        dic[target] = 1

print(len(dic.keys()))