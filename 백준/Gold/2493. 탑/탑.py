import sys, heapq

N = int(sys.stdin.readline().strip())
towers = list(map(int, (sys.stdin.readline()).split()))
result = []

q = []
for i in range(N - 1, -1, -1) :
    while q and towers[i] > q[0][0] :
        result.append([i + 1, heapq.heappop(q)[1]])
    heapq.heappush(q, [towers[i], i + 1])

for i in q :
    result.append([0, i[1]])

result = sorted(result, key=lambda x : x[1])
for a, b in result :
    print(a, end=" ")