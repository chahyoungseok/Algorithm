def find_parent(parent, x):
    if parent[x] != x:
        return find_parent(parent, parent[x])
    return parent[x]


def union_parent(parent, a, b):
    a = find_parent(parent, a)
    b = find_parent(parent, b)

    if a > b:
        parent[a] = b
    else:
        parent[b] = a


def solution(n, costs):
    answer = 0
    parent = [i for i in range(n)]

    costs = sorted(costs, key=lambda x: x[2])

    for case in costs:
        land_1, land_2, cost = case
        if find_parent(parent, land_1) != find_parent(parent, land_2):
            union_parent(parent, land_1, land_2)
            answer += cost

    return answer