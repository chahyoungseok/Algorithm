import sys


def init(node, start, end) :
    if start == end :
        tree[node] = table[start]
    else :
        mid = (start + end) // 2
        tree[node] = (init(node * 2, start, mid) * init(node * 2 + 1, mid + 1, end)) % 1000000007

    return tree[node]


def sub_sum(node, start, end, left, right) :
    if start > right or end < left :
        return 1

    if left <= start and end <= right :
        return tree[node]

    mid = (start + end) // 2
    return sub_sum(node * 2, start, mid, left, right) * sub_sum(node * 2 + 1, mid + 1, end, left, right)  % 1000000007


def update(node, index, value, start, end) :
    if index < start or index > end :
        return

    if start == end :
        tree[node] = value
        return

    mid = (start + end) // 2
    update(node * 2, index, value, start, mid)
    update(node * 2 + 1, index, value, mid + 1, end)
    tree[node] = tree[node * 2] * tree[node * 2 + 1] % 1000000007


N, M, K = map(int, (sys.stdin.readline()).split())

table = []
tree = [0 for _ in range(N * 4)]

for _ in range(N) :
    table.append(int(sys.stdin.readline().strip()))

init(1, 0, N - 1)

for _ in range(M + K) :
    a, b, c = map(int, (sys.stdin.readline()).split())

    if a == 1 :
        b -= 1
        table[b] = c
        update(1, b, c, 0, N - 1)
    else :
        print(int(sub_sum(1, 0, N - 1, b - 1, c - 1)) % 1000000007)