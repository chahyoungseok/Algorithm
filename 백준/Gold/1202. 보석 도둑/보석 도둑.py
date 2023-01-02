import sys, heapq
from collections import deque

N, K = map(int, (sys.stdin.readline()).split())
jewel, bags = [], []

for _ in range(N) :
    jewel.append(list(map(int, (sys.stdin.readline()).split())))

for _ in range(K) :
    bags.append(int(sys.stdin.readline().strip()))

jewel.sort()
jewel = deque(jewel)
bags.sort()

q, total = [], 0
for i in bags :
    while jewel and i >= jewel[0][0] :
        heapq.heappush(q, -jewel.popleft()[1])
    if q :
        total += -heapq.heappop(q)

print(total)