import sys


def init(node, start, end) :
    if start == end :
        tree[node] = l[start]
    else :
        mid = (start + end) // 2
        tree[node] = init(node * 2, start, mid) + init(node * 2 + 1, mid + 1, end)
    return tree[node]


def subSum(node, start, end, left, right) :
    if start > right or end < left :
        return 0

    if left <= start and end <= right :
        return tree[node]

    mid = (start + end) // 2
    return subSum(node * 2, start, mid, left, right) + subSum(node * 2 + 1, mid + 1, end, left, right)


def update(node, start, end, index, diff) :
    if start > index or end < index :
        return

    tree[node] += diff

    if start != end :
        mid = (start + end) // 2
        update(node * 2, start, mid, index, diff)
        update(node * 2 + 1, mid + 1, end, index, diff)


N, M, K = map(int, (sys.stdin.readline()).split())
l = []
for _ in range(N) :
    l.append(int(sys.stdin.readline().strip()))

tree = [0] * (N *10)
init(1, 0, N - 1)

for _ in range(M + K) :
    a, b, c = map(int, (sys.stdin.readline()).split())
    if a == 1 :
        diff = c - l[b - 1]
        l[b - 1] = c
        update(1, 0, N - 1, b - 1, diff)
    else :
        print(subSum(1, 0, N - 1, b - 1, c - 1))