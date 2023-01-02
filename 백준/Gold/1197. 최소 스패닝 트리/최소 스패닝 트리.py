# 가르침
# 최소 스패닝 트리
# 부분 문자열
# 최단거리 음수
import sys, heapq


def find_parent(parent, x) :
    if parent[x] != x :
        return find_parent(parent, parent[x])
    return x


def union_parent(parent, a, b) :
    a = find_parent(parent, a)
    b = find_parent(parent, b)

    if a > b :
        parent[b] = a
    else :
        parent[a] = b


V, E = map(int, (sys.stdin.readline()).split())

q, total, parent = [], 0,  [i for i in range(V + 1)]
for _ in range(E) :
    A, B, C = map(int, (sys.stdin.readline()).split())
    heapq.heappush(q, [C, A, B])

while q :
    C, A, B = heapq.heappop(q)
    if find_parent(parent, A) != find_parent(parent, B) :
        union_parent(parent, A, B)
        total += C

print(total)