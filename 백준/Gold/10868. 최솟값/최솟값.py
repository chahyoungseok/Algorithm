import sys


def init(node, start, end) :
    if start == end :
        tree[node] = l[start]
    else :
        mid = (start + end) // 2
        tree[node] = min(init(node * 2, start, mid), init(node * 2 + 1, mid + 1, end))
    return tree[node]


def search(node, start, end, left, right) :
    if start > right or end < left :
        return int(1e12)

    if left <= start and end <= right :
        return tree[node]

    mid = (start + end) // 2
    return min(search(node * 2, start, mid, left, right), search(node * 2 + 1, mid + 1, end, left, right))


N, M = map(int, (sys.stdin.readline()).split())
l = []
tree = [0] * (4 * N)
for _ in range(N) :
    l.append(int(sys.stdin.readline().strip()))

init(1, 0, N - 1)
for _ in range(M) :
    a, b = map(int, (sys.stdin.readline()).split())
    print(search(1, 0, N - 1, a - 1, b - 1))