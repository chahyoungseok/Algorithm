import sys, heapq

N = int(sys.stdin.readline().strip())

left, right, mid = [], [], int(sys.stdin.readline().strip())
left_count, right_count = 0, 0
print(mid)
for i in range(1, N) :
    data = int(sys.stdin.readline().strip())
    if mid > data :
        heapq.heappush(left, -data)
        left_count += 1
    else :
        heapq.heappush(right, data)
        right_count += 1
    if left_count > right_count :
        right_count += 1
        left_count -= 1
        heapq.heappush(right, mid)
        mid = -heapq.heappop(left)
    if right_count > left_count + 1 :
        right_count -= 1
        left_count += 1
        heapq.heappush(left, -mid)
        mid = heapq.heappop(right)
    print(mid)