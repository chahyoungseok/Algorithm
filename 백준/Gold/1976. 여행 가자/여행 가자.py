import sys


def find_parent(x):
    if parent[x] != x:
        parent[x] = find_parent(parent[x])
    return parent[x]


def union_parent(a, b):
    if a == b:
        return

    a = find_parent(a)
    b = find_parent(b)
    if a > b :
        parent[a] = b
    else:
        parent[b] = a


N = int(sys.stdin.readline().strip())
M = int(sys.stdin.readline().strip())

parent = [ele for ele in range(N + 1)]

for host_node in range(1, N + 1):
    edge = list(map(int, sys.stdin.readline().strip().split()))

    for node in range(1, N + 1):
        if edge[node - 1] == 1:
            union_parent(host_node, node)

result = "YES"
trip = list(map(int, sys.stdin.readline().strip().split()))
depart_city = find_parent(trip[0])
for city in range(1, len(trip)):
    if find_parent(trip[city]) != depart_city :
        result = "NO"
        break

print(result)