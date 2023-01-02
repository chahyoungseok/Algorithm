import sys


def init(node, start, end) :
    if start == end :
        tree[node] = N_list[start]
    else:
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
    if index < start or index > end :
        return

    tree[node] += diff
    if start != end :
        mid = (start + end) // 2
        update(node * 2, start, mid, index, diff)
        update(node * 2 + 1, mid + 1, end, index, diff)


N, Q = map(int, (sys.stdin.readline()).split())
N_list = list(map(int, (sys.stdin.readline()).split()))

tree = [0] * (N * 5)
init(1, 0, N - 1)

for _ in range(Q) :
    x, y, a, b = map(int, (sys.stdin.readline()).split())
    if x > y :
        print(subSum(1, 0, N - 1, y - 1, x - 1))
    else :
        print(subSum(1, 0, N - 1, x - 1, y - 1))

    a = a - 1
    diff = b - N_list[a]
    N_list[a] = b
    update(1, 0, N - 1, a, diff)