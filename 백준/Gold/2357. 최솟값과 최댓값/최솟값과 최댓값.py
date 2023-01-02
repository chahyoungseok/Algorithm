import sys


def init(node, start, end) :
    if start == end :
        tree[node] = [l[start], l[start]]
    else :
        mid = (start + end) // 2
        left, right = init(node * 2, start, mid), init(node * 2 + 1, mid + 1, end)
        tree[node] = [min(left[0], right[0]), max(left[1], right[1])]
    return tree[node]


def search(node, start, end, left, right) :
    if start > right or end < left :
        return [int(1e11), 0]

    if left <= start and end <= right :
        return tree[node]

    mid = (start + end) // 2
    le, ri = search(node * 2, start, mid, left, right), search(node * 2 + 1, mid + 1, end, left, right)
    return [min(le[0], ri[0]), max(le[1], ri[1])]


N, M = map(int, (sys.stdin.readline()).split())
l = []
tree = [0] * (4 * N)
for _ in range(N) :
    l.append(int(sys.stdin.readline().strip()))

init(1, 0, N - 1)
for _ in range(M) :
    a, b = map(int, (sys.stdin.readline()).split())
    print(*search(1, 0, N - 1, a - 1, b - 1))