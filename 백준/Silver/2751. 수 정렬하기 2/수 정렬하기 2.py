import sys, heapq

N = int(input())
q = []
for _ in range(N) :
    heapq.heappush(q, int(sys.stdin.readline()))

for _ in range(N) :
    print(heapq.heappop(q))